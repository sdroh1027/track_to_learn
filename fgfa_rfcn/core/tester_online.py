from multiprocessing.pool import ThreadPool as Pool
import cPickle
import os
import time
import mxnet as mx
import numpy as np
import dill
from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.seq_nms import seq_nms
from utils.PrefetchingIter import PrefetchingIter
from collections import deque

from fgfa_rfcn.core.instance import Instance
from fgfa_rfcn.core.tester import get_resnet_output, draw_all_detection, prepare_data
import logging
logger = logging.getLogger("lgr")

def roi_crop_embed(img_height, img_width, pred_boxes, cur_embed, cfg):
    # using cur_embed & bbox_pred, make new instance wise embed layer
    # bbox_pred[4:8] has (x1,y1) (x2,y2) coordinates

    scale_h = cur_embed.shape[2] / img_height
    scale_w = cur_embed.shape[3] / img_width

    zeros = np.zeros([300, 1])
    pred_boxes_on_feat = np.zeros([300, 4])
    pred_boxes_on_feat[:, 0] = pred_boxes[:, 4] * scale_w  # x1
    pred_boxes_on_feat[:, 1] = pred_boxes[:, 5] * scale_h  # y1
    pred_boxes_on_feat[:, 2] = pred_boxes[:, 6] * scale_w  # x2
    pred_boxes_on_feat[:, 3] = pred_boxes[:, 7] * scale_h  # y2
    pred_boxes_on_feat = clip_boxes(pred_boxes_on_feat, cur_embed.shape[-2:])
    pred_boxes_on_feat = np.hstack((zeros, pred_boxes_on_feat[:, 0:]))
    pred_boxes_on_feat = mx.nd.array(pred_boxes_on_feat, mx.gpu())

    sym_bbox = mx.sym.Variable('bbox_with_delta')
    sym_cur_embed = mx.sym.Variable('cur_embed')
    new_embed = mx.sym.ROIPooling(data=sym_cur_embed, rois=sym_bbox, pooled_size=(cfg.TEST.EMBED_SIZE, cfg.TEST.EMBED_SIZE),
                                  spatial_scale=1.0, name='new_embed')
    new_embed_normalized = mx.sym.L2Normalization(data=new_embed, mode='channel', name='new_embed_normalized')
    ex = new_embed_normalized.bind(mx.gpu(), {'cur_embed': cur_embed, 'bbox_with_delta': pred_boxes_on_feat})
    cropped_embed = ex.forward()

    return cropped_embed


def im_detect_all(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)
    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    iscores_all = [] # instance box prediction
    ipred_boxes_all = [] # instance box prediction
    rois_all = []
    #cur_embeds_all = [] #
    #new_embeds_all = [] #
    #sliced_all = [] #
    cropped_embeds_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        iscores = output['inst_prob_reshape_output'].asnumpy()[0]
        ibbox_deltas = output['ibbox_pred_reshape_output'].asnumpy()[0]
        cur_embed = output['cur_embed_output']
        #unnormalize_weight = output['unnormalize_weight_output']
        #new_embed = output['new_embed_output'].asnumpy()[0]
        #sliced = output['sliced_bbox_output'].asnumpy()[0]

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:]) # Clip boxes to image boundaries.
        ipred_boxes = bbox_pred(rois, ibbox_deltas)
        ipred_boxes = clip_boxes(ipred_boxes, im_shape[-2:])

        # for cropping, scale pred boxes to size of embed feature
        img_height =data_batch.data[0][1][0][0].asnumpy()
        img_width =data_batch.data[0][1][0][1].asnumpy()
        cropped_embed = roi_crop_embed(img_height, img_width, pred_boxes, cur_embed, cfg)
        cropped_embed = cropped_embed[0].asnumpy()

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale
        ipred_boxes = ipred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
        iscores_all.append(iscores)
        ipred_boxes_all.append(ipred_boxes)
        rois_all.append(rois)
        #cur_embeds_all.append(cur_embed)
        #new_embeds_all.append(new_embed)
        #sliced_all.append(sliced)
        cropped_embeds_all.append(cropped_embed)

        debug = 0
        if debug is True:
            cur_embed = cur_embed.asnumpy()[0]
            feat_cache = output['feat_cache'].asnumpy()[0]
            #plot_tensor(cur_embed, 16)
            #plot_tensor(feat_cache, 16)

    return zip(scores_all, pred_boxes_all, rois_all, data_dict_all, iscores_all, ipred_boxes_all, cropped_embeds_all)


def pred_eval_ot(gpu_id, feat_predictors, aggr_predictors, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    det_file = os.path.join(imdb.result_path, imdb.name + '_'+ str(gpu_id))
    if cfg.TEST.SEQ_NMS == True:
        det_file += '_raw'
    print 'det_file=',det_file
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            all_boxes, frame_ids = cPickle.load(fid)
        return all_boxes, frame_ids


    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]
    num_images = test_data.size
    roidb_frame_ids = [x['frame_id'] for x in test_data.roidb]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)
    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    frame_ids = np.zeros(num_images, dtype=np.int)

    roidb_idx = -1
    roidb_offset = -1
    idx = 0
    all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1

    data_time, net_time, post_time,seq_time = 0.0, 0.0, 0.0,0.0
    t = time.time()

    # loop through all the test data
    for im_info, key_frame_flag, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        #################################################
        # new video                                     #
        #################################################
        # empty lists and append padding images
        # do not do prediction yet
        if key_frame_flag == 0:
            roidb_idx += 1
            roidb_offset = -1
            # init data_lsit and feat_list for a new video
            data_list = deque(maxlen=all_frame_interval)
            feat_list = deque(maxlen=all_frame_interval)
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            # append cfg.TEST.KEY_FRAME_INTERVAL+1 padding images in the front (first frame)
            while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL+1:
                data_list.append(image)
                feat_list.append(feat)

            ginst_mem = []  # list for instance class
            sim_array_global = []  # similarity array list
            ginst_ID = 0

        #################################################
        # main part of the loop                         #
        #################################################
        elif key_frame_flag == 2:
            # keep appending data to the lists without doing prediction until the lists contain 2 * cfg.TEST.KEY_FRAME_INTERVAL objects
            if len(data_list) < all_frame_interval - 1:
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

            else:
                scales = [iim_info[0, 2] for iim_info in im_info]

                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                pred_result = im_detect_all(aggr_predictors, data_batch, data_names, scales, cfg)

                roidb_offset += 1 # frame number in this snippet
                frame_ids[idx] = roidb_frame_ids[roidb_idx] + roidb_offset

                if frame_ids[idx] == 7000:
                    print 'stop point'

                t2 = time.time() - t
                t = time.time()
                ginst_ID, out_im, out_im2 = process_pred_result_ot(imdb.classes, pred_result, imdb.num_classes, thresh, cfg, nms, all_boxes,
                                                         roidb_offset, max_per_image, vis,
                                                         data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales,
                                                         ginst_mem, sim_array_global, ginst_ID)
                idx += test_data.batch_size

                t3 = time.time() - t
                t = time.time()
                data_time += t1
                net_time += t2
                post_time += t3
                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images,
                                                                                      data_time / idx * test_data.batch_size,
                                                                                      net_time / idx * test_data.batch_size,
                                                                                      post_time / idx * test_data.batch_size)
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size))
        #################################################
        # end part of a video                           #
        #################################################
        elif key_frame_flag == 1:       # last frame of a video
            end_counter = 0
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                data_list.append(image)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                pred_result = im_detect_all(aggr_predictors, data_batch, data_names, scales, cfg)

                roidb_offset += 1
                frame_ids[idx] = roidb_frame_ids[roidb_idx] + roidb_offset

                t2 = time.time() - t
                t = time.time()
                ginst_ID, out_im, out_im2 = process_pred_result_ot(imdb.classes, pred_result, imdb.num_classes, thresh, cfg, nms,
                                                                   all_boxes, roidb_offset, max_per_image, vis,
                                                                   data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(),
                                                                   scales, ginst_mem, sim_array_global, ginst_ID)
                idx += test_data.batch_size
                t3 = time.time() - t
                t = time.time()
                data_time += t1
                net_time += t2
                post_time += t3

                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images,
                                                                                   data_time / idx * test_data.batch_size,
                                                                                   net_time / idx * test_data.batch_size,
                                                                                   post_time / idx * test_data.batch_size)
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size))
                end_counter += 1

    with open(det_file, 'wb') as f:
        cPickle.dump((all_boxes, frame_ids), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return all_boxes, frame_ids


def pred_eval_multiprocess_ot(gpu_num, key_predictors, cur_predictors, test_datas, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):

    assert cfg.TEST.SEQ_NMS==False
    if gpu_num == 1:
        res = [pred_eval_ot(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis, thresh, logger,
                         ignore_cache), ]
    else:
        #multigpu not supported yet
        assert gpu_num == 1

        from multiprocessing.pool import ThreadPool as Pool
        pool = Pool(processes=gpu_num)
        multiple_results = [pool.apply_async(pred_eval_ot, args=(
        i, key_predictors[i], cur_predictors[i], test_datas[i], imdb, cfg, vis, thresh, logger, ignore_cache)) for i
                            in range(gpu_num)]
        pool.close()
        pool.join()
        res = [res.get() for res in multiple_results]
    info_str = imdb.evaluate_detections_multiprocess(res)

    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))


def draw_all_instances(im_array, inst_mem_now, inst_mem, class_names, scale, cfg, threshold=0.1): # sidong
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param inst_mem_now: local instance array
    :param inst_mem: global instance array
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :param IDs: linked  global IDs of local instances. tuple.
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, inst in enumerate(inst_mem_now):
        if inst.cls == 0: # '__background__':
            continue
        if len(inst.linked_to) == 0:
            continue
        ID = inst.linked_to  # note that this is tuple.
        if not inst_mem[ID[0]].color:
            inst_mem[ID[0]].color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        bbox = inst.bbox[:4] * scale
        cls = inst.cls
        score = inst.cls_score
        #if score < threshold:
        #    continue
        bbox = map(int, bbox)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=inst_mem[ID[0]].color, thickness=2)
        cv2.putText(im, 'LID:%d, %s %.2f, GID:%s, %s, dc:%.2f' % (inst.LID, class_names[cls], score, ID, class_names[inst_mem[ID[0]].cls_high], inst_mem[ID[0]].cls_score_decayed), (bbox[0], bbox[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def divide_L2_norm(vector): #assume that the vector is 2-demensional list (2048L, cfg.TEST.EMBED_SIZE, cfg.TEST.EMBED_SIZE)
    return vector / np.linalg.norm(vector, axis=0) # vector <type 'tuple'>: (2048, 8,8)

def compare_embed(A, B):
    C = A * B # C vector: <type 'tuple'>: (2048, 8,8)
    sum = np.sum(C, axis=0)  # return vector: <type 'tuple'>: (8,8)
    sum1 = np.sum(sum, axis=0)  # return vector: <type 'tuple'>: (8)
    sum2 = np.sum(sum1, axis=0)  # return vector: <type 'tuple'>: (1)
    return sum2/64

def Area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def compute_IOU(box_A,box_B):
    area_A = Area(box_A)
    area_B = Area(box_B)

    xx1 = max(box_A[0],box_B[0])
    yy1 = max(box_A[1],box_B[1])
    xx2 = min(box_A[2],box_B[2])
    yy2 = min(box_A[3],box_B[3])

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (area_A + area_B - inter)
    return ovr


def draw_all_rois(im_array, rois, scale, cfg, threshold=0.1): # sidong
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color

    #print rois
    for det in rois:
        bbox = det[:4] #* scale # scaling is not needed for rois
        #score = det[-1]
        #if score < threshold:
        #    continue
        bbox = map(int, bbox)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        #cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
        #            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        cv2.putText(im, '%s' % ('rois'), (bbox[0], bbox[1] + 10),
                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im



def process_pred_result_ot(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, center_image, scales, inst_mem, sim_array_global, ginst_ID):
    global logger

    for delta, (scores, boxes, rois, data_dict, iscores, ipred_boxes, embed_feat) in enumerate(pred_result): #16th frame -> cat(10) -> local box(3, 4th in rois)
        local_inst_ID = 0
        inst_mem_now = [] # intra-frame inst memory

        for j  in range(1,num_classes):
            logger.info('[%dth frame]test boxes of class: (%d, %s)' % (idx+delta, j, classes[j]))
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][idx+delta]=cls_dets # all_boxes[31L(class)][frame_idx + batch_size][300L(box_idx)][5L(x,y,w,h,score)]
            else:
                cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]

            # intra frame supression
            #print 'intra frame phase, class:%d(%s)' % (j,classes[j])
            #qwe = len(all_boxes[j][idx + delta])
            #zxc = range(len(all_boxes[j][idx + delta]))
            for box_num in range(len(all_boxes[j][idx+delta])): # for output boxes of class j,
                new_inst = Instance(LID=local_inst_ID, frame_idx=idx + delta, cls=j, cls_score=cls_scores[box_num], bbox=cls_boxes[box_num],  embed_feat=embed_feat[box_num]) #make instances for outputs
                #logger.info('[%dth frame] %dth %s box(%.3f)' % (idx + delta, box_num, classes[j], new_inst.cls_score))
                is_new = 1
                if not inst_mem_now:  # if inst mem is empty,
                    logger.info('[%dth frame] %dth %s box(%.3f) is first one of local inst_memory (new_ID:%d)' % (
                    idx + delta, box_num, classes[j], new_inst.cls_score, local_inst_ID))
                    inst_mem_now.append(new_inst)
                    local_inst_ID += 1
                else:  # if local mem is not empty, comparte candidate box with local mem
                    for inst in inst_mem_now:
                        if is_new < 1 :
                            break
                        assert inst.detected_idx[-1] == new_inst.detected_idx[-1] # two objects must be in the same frame, highest score class is best
                        #temp_inst = []
                        coeff = compare_embed(inst.embed_feat, new_inst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                        iou = compute_IOU(inst.bbox,new_inst.bbox)
                        coeff_th = (coeff > cfg.TEST.COEFF_THRESH_INTER)
                        iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                        similar = iou * coeff
                        similar_th = (similar > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
                        logger.debug( '[%dth frame] %dth %s box(%.3f) VS local_inst_ID(%d) (iou:%.3f(%s), coeff:%.3f(%s), sim:%.3f(%s))' % (
                            idx + delta, box_num, classes[j], new_inst.cls_score, inst.LID, iou, iou_th, coeff, coeff_th, similar, similar_th))

                        if similar > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA:  # simillar
                            is_new -= 1
                            logger.info('[%dth frame] %dth %s box(%.3f) is simillar with local_inst(ID:%d, class:%s, frame:%d)' % (
                                idx + delta, box_num, classes[j], new_inst.cls_score, inst.LID, classes[inst.cls],
                                inst.detected_idx[-1]))
                            inst.update_intra_frame(new_inst)
                        elif similar <= cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA:  # not similar
                            continue
                        else:  # if coeff is Nan or something
                            logger.error('error: coeff value is not normal')


                    if int(is_new) is 1:
                        logger.info('[%dth frame] %dth %s box(%.3f) is added as a new local inst(new_ID:%d)' % (
                            idx + delta, box_num, classes[j], new_inst.cls_score, local_inst_ID))
                        inst_mem_now.append(new_inst)
                        local_inst_ID += 1
                    else:
                        logger.info('[%dth frame] %dth %s box(%.3f) is not a new local inst' % (
                            idx + delta, box_num, classes[j], new_inst.cls_score))

        sim_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final_th = np.zeros((1,len(inst_mem_now)), int)

        # Similarity matrix generation (global <-> local inst)
        logger.info('[%dth frame] @@@ inter frame phase start @@@', idx + delta)
        if len(inst_mem) > 0 and len(inst_mem_now) > 0 :
            sim_array = np.zeros((len(inst_mem), len(inst_mem_now))) # matrix of similarity
            index_sim_max = np.zeros(len(inst_mem), dtype= int)
            sim_sort_index = np.zeros((len(inst_mem), len(inst_mem_now)))
            for i, ginst in enumerate(inst_mem):  # loop for instances in this frame
                for j, linst in enumerate(inst_mem_now):
                    coeff = compare_embed(ginst.embed_feat,
                                          linst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                    iou = compute_IOU(ginst.bbox2, linst.bbox2) #use virtual bbox
                    similar = coeff * iou
                    coeff_th = (coeff > cfg.TEST.COEFF_THRESH_INTER)
                    iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                    similar_th = ( coeff * iou > cfg.TEST.IOU_THRESH_INTRA * cfg.TEST.COEFF_THRESH_INTRA)
                    sim_array[i, j] = similar
                    logger.debug('[%dth frame] global_inst[%d](cls:%s, %.3f, last_frame:%s)) VS local_inst[%d](cls:%s, %.3f): (iou:%.3f(%s), coeff:%.3f(%s), sim:%.3f(%s))' % (
                        idx + delta, i, classes[ginst.cls], ginst.cls_score, ginst.detected_idx[-1], j, classes[linst.cls], linst.cls_score,
                        iou, iou_th, coeff, coeff_th, similar, similar_th))

                sim_sort_index[i] = np.argsort(sim_array[i]) # sim_sort_index[i][-1] means most similar one

            sim_array_th = (sim_array[:][:] > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
            sim_array_global.append(sim_array) #for debugging

            # Conflict handling (two or more ginst simillar to same linst)
            # TODO: delete this
            #gid_to_supress_all = set() # set deletes redundant values
            #for id, ginst in enumerate(inst_mem):
            #    gids = np.where(sim_array_th[:][index_sim_max[id]] == True)[0] #id of ginsts which wants same linsts
            #    if len(gids) == 1:
            #        continue
            #    elif len(gids) > 1: #conflict occurs
            #        values = sim_array[gids, index_sim_max[id]]
            #        idx_me = np.where(gids == id)[0]
            #        if np.argmax(values) == idx_me: # search if this gid has biggest score
            #            #TODO: supress others
            #            del gids[idx_me]
            #            gid_to_supress = gids
            #            gid_to_supress_all.add(gid_to_supress)
            #            inst_mem[gid_to_supress].gid_suppressed_by = id
            #        else:
            #            index_sim_max[id] = -1
            #    else :
            #        logger.error('conflict handling error!!!')

            #linst guided ginst suppresion
            sim_array2 = sim_array.copy()
            sim_array2_th = sim_array_th.copy() * int(1) # bool to int
            for i, linst in enumerate(inst_mem_now):
                lindex_over_th = np.where(sim_array2_th[:,i] == True)[0]
                if len(lindex_over_th) > 1:
                    lindex_to_suppress = lindex_over_th[:]
                    survive = lindex_to_suppress.min()
                    lindex_to_suppress = np.setdiff1d(lindex_to_suppress,survive) # oldest ginst(me) survives
                    if sim_array2_th[survive, i] != -1:  # if not suppressed, suppress others
                        sim_array2[lindex_to_suppress, :] = -1
                        sim_array2_th[lindex_to_suppress, :] = -1

            # ginst guided linst supression
            sim_array3 = sim_array2.copy()
            sim_array3_th = sim_array2_th.copy()
            for i, ginst in enumerate(inst_mem):
                gindex_over_th = np.where(sim_array3_th[i] == True)[0]
                if len(gindex_over_th) > 1:
                    index_max = np.argmax(sim_array3[i])
                    gindex_to_suppress = gindex_over_th[:]
                    gindex_to_suppress = np.setdiff1d(gindex_to_suppress,index_max)
                    if sim_array3_th[i][index_max] != -1: #if not suppressed
                        sim_array3[:, gindex_to_suppress] = -1
                        sim_array3_th[:, gindex_to_suppress] = -1

            sim_array_final = sim_array3
            sim_array_final_th = sim_array3_th

            if idx == 107: # 115
                print 'debug_point'

            # linking global inst to local insts
            for i, ginst in enumerate(inst_mem):
                index_sim_max[i] = np.argmax(sim_array_final[i])
                if index_sim_max[i] == -1:
                    continue
                max_coeff = sim_array_final[i][index_sim_max[i]]
                if max_coeff >= (cfg.TEST.IOU_THRESH_INTER * cfg.TEST.COEFF_THRESH_INTER):
                    #inst_mem_now[sim_max_index[i]].linked_to.append(ginst.GID)
                    logger.info(
                        '[%dth frame] global inst[%d](cls:%s, %.3f, frame:%d) is updated by local box[%d](cls:%s, %.3f)' % (
                            idx + delta, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                            inst_mem[i].detected_idx[-1], index_sim_max[i], classes[inst_mem_now[index_sim_max[i]].cls],
                            inst_mem_now[index_sim_max[i]].cls_score))
                    ginst.update_inter_frame2(inst_mem_now[index_sim_max[i]], max_coeff)
                else:
                    logger.info('global inst[%d](cls:%s, %.3f, frame:%d) is not linked to anyone' % (
                        inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score, inst_mem[i].detected_idx[-1]))

        #link to previoust ginst or make new ginst for each local insts

        append_new_insts = []
        if len(inst_mem_now) > 0:
            for j, linst in enumerate(inst_mem_now):
                if sim_array_final[0][j] == -1:
                    logger.info(
                    '[%dth frame] local_inst[%d](cls:%s, %.3f) is discarded because of overlap' % (
                        idx + delta, j, classes[linst.cls], linst.cls_score))
                    continue
                if len(linst.linked_to) == 0 :
                    if linst.cls_score > cfg.TEST.SCORE_THRESH:
                        logger.info('[%dth frame] local_inst[%d](cls:%s, %.3f) is added as global inst_memory[%d]' % (
                            idx + delta, j, classes[linst.cls], linst.cls_score, ginst_ID))
                        new_ginst = linst.make_global_inst(ginst_ID)
                        append_new_insts.append(new_ginst)
                        linst.linked_to = [ginst_ID]
                        ginst_ID = ginst_ID + 1
                    else:
                        logger.info(
                            '[%dth frame] local_inst[%d](cls:%s, %.3f) is discarded because of low cls_score' % (
                                idx + delta, j, classes[linst.cls], linst.cls_score))

        inst_mem.extend(append_new_insts)

        out_im = 0
        out_im2 = 0
        if vis:
            out_im2 = draw_all_instances(center_image, inst_mem_now, inst_mem, classes, scales[delta], cfg)
            #out_im2 = draw_all_rois(center_image, rois, scales[delta], cfg) # print rois from RPN

        if cfg.TEST.SEQ_NMS==False and  max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][idx + delta][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][idx + delta][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx + delta] = all_boxes[j][idx + delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, num_classes)]
                out_im = draw_all_detection(center_image, boxes_this_image, classes, scales[delta], cfg)

            return ginst_ID, out_im, out_im2

    return ginst_ID, out_im, out_im2
