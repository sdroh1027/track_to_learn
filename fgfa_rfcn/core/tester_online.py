from multiprocessing.pool import ThreadPool as Pool
import cPickle
import os
import time
import mxnet as mx
import numpy as np
import math
import dill
from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.seq_nms import seq_nms
from utils.PrefetchingIter import PrefetchingIter
from collections import deque

from instance import Instance
from tester import get_resnet_output, draw_all_detection, prepare_data
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
    #new_embed = mx.sym.ROIPooling(data=sym_cur_embed, rois=sym_bbox, pooled_size=(cfg.TEST.EMBED_SIZE, cfg.TEST.EMBED_SIZE),
    #                              spatial_scale=1.0, name='new_embed')
    new_embed = mx.sym.contrib.ROIAlign(data=sym_cur_embed, rois=sym_bbox,
                                  pooled_size=(cfg.TEST.EMBED_SIZE, cfg.TEST.EMBED_SIZE),
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
    psroipooled_cls_rois_all = []
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
        psroipooled_cls_rois = output['psroipooled_cls_rois_output'].asnumpy()
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
        psroipooled_cls_rois_all.append(psroipooled_cls_rois)

        debug = 0
        if debug is True:
            cur_embed = cur_embed.asnumpy()[0]
            feat_cache = output['feat_cache'].asnumpy()[0]
            #plot_tensor(cur_embed, 16)
            #plot_tensor(feat_cache, 16)

    return zip(scores_all, pred_boxes_all, rois_all, data_dict_all, iscores_all, ipred_boxes_all, cropped_embeds_all, psroipooled_cls_rois_all)


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
    all_boxes_inst = [[[] for _ in range(num_images)]
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

            ginst_ID = 0
            ginst_mem = []  # list for instance class
            sim_array_global = []  # similarity array list
            logger.info('prepared for a new video')
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

                t2 = time.time() - t
                t = time.time()
                ginst_ID_prev = ginst_ID
                ginst_ID, out_im, out_im2, out_im_linst = process_link_pred_result(imdb.classes, pred_result, imdb.num_classes, thresh,
                                                                     cfg, nms, all_boxes, all_boxes_inst, idx,
                                                                     max_per_image, vis, data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales,
                                                                     ginst_mem, sim_array_global, ginst_ID)
                ginst_ID_now = ginst_ID
                idx += test_data.batch_size

                t3 = time.time() - t
                t = time.time()
                data_time += t1
                net_time += t2
                post_time += t3
                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s #ginst:{}'.format(idx, num_images,
                                                                                      data_time / idx * test_data.batch_size,
                                                                                      net_time / idx * test_data.batch_size,
                                                                                      post_time / idx * test_data.batch_size, ginst_ID)
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s #ginst:{}'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size, ginst_ID))
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
                ginst_ID_prev = ginst_ID
                ginst_ID, out_im, out_im2, out_im_linst = process_link_pred_result(imdb.classes, pred_result, imdb.num_classes, thresh, cfg, nms,
                                                                   all_boxes, all_boxes_inst, idx, max_per_image, vis,
                                                                   data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(),
                                                                   scales, ginst_mem, sim_array_global, ginst_ID)
                ginst_ID_now = ginst_ID
                idx += test_data.batch_size
                t3 = time.time() - t
                t = time.time()
                data_time += t1
                net_time += t2
                post_time += t3

                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s #ginst:{}'.format(idx, num_images,
                                                                                   data_time / idx * test_data.batch_size,
                                                                                   net_time / idx * test_data.batch_size,
                                                                                   post_time / idx * test_data.batch_size, ginst_ID)
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s #ginst:{}'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size, ginst_ID))
                end_counter += 1

    with open(det_file, 'wb') as f:
        cPickle.dump((all_boxes, frame_ids), f, protocol=cPickle.HIGHEST_PROTOCOL)

    return all_boxes_inst, frame_ids


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

def draw_all_ginst(im_array, inst_mem, class_names, scale, cfg, frame_now, threshold=0.1): # sidong
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
    for j, inst in enumerate(inst_mem):
        if inst.cls == 0: # '__background__':
            continue
        if inst.detected_idx[-1] == frame_now:
            if not inst.color:
                inst.color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
            bbox = inst.bbox[:4] * scale
            cls = inst.cls
            score = inst.cls_score
            #if score < threshold:
            #    continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=inst.color, thickness=2)
            cv2.putText(im, 'LID:%d %.3s %.2f GID:%s %.3s r:%.2f' % (inst.LID, class_names[cls], score, inst.GID,
                        class_names[inst.cls_high], inst.cls_score_reliable),
                        (bbox[0], bbox[1] + 10), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_all_linst(im_array, inst_mem_now, inst_mem, class_names, scale, cfg, threshold=0.1): # sidong
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
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        bbox = inst.bbox[:4] * scale
        cls = inst.cls
        score = inst.cls_score
        #if score < threshold:
        #    continue
        bbox = map(int, bbox)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        cv2.putText(im, 'LID:%d %.3s %.2f' % (inst.LID, class_names[cls], score),
                    (bbox[0], bbox[1] + 10), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
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

def compute_dist(inst_A, inst_B):
    rad_A = math.sqrt(Area(inst_B.bbox))
    rad_B = math.sqrt(Area(inst_B.bbox))
    dx = inst_A.center[0] - inst_B.center[0]
    dy = inst_A.center[1] - inst_B.center[1]
    dis = math.sqrt(dx ** 2 + dy ** 2)
    relative_dist = ((rad_A + rad_B) - dis)/(rad_A + rad_B)
    return (0 if relative_dist <= 0 else relative_dist)


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



def process_link_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, all_boxes_inst, idx,
                             max_per_image, vis, center_image, scales, inst_mem, sim_array_global, ginst_ID):
    global logger

    for delta, (scores, boxes, rois, data_dict, iscores, ipred_boxes, embed_feat, psroipooled_cls_rois) in enumerate(pred_result): #16th frame -> cat(10) -> local box(3, 4th in rois)
        local_inst_ID = 0
        inst_mem_now = [] # intra-frame inst memory

        per_cls_box_idx_over_th = [[]]
        per_cls_box_idx_nms = [[]]
        for j in range(1, num_classes):
            #logger.debug('[%dth frame]test boxes of class: (%d, %s)' % (idx+delta, j, classes[j]))
            indexes = np.where(scores[:, j] > thresh)[0]
            per_cls_box_idx_over_th.append(indexes)
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][idx + delta]=cls_dets # all_boxes[31L(class)][frame_idx + batch_size][300L(box_idx)][5L(x,y,w,h,score)]
            else:
                cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]
                per_cls_box_idx_nms.append(indexes[keep])

        # intra frame supression (class wise)
        for j in range(1, num_classes):
            inst_mem_cls = [[] for _ in range(num_classes)] ## intra-class inst memory
            box_indexes = per_cls_box_idx_nms[j]
            for i, box_idx in enumerate(box_indexes): # for dets of class j,
                box = all_boxes[j][idx + delta][i]
                new_inst = Instance(LID=local_inst_ID, frame_idx=idx + delta, cls=j, cls_score=box[-1],
                                    bbox=box[0:4],  embed_feat=embed_feat[box_idx],
                                    atten=psroipooled_cls_rois[box_idx]) #make instances for outputs
                local_inst_ID += 1
                is_new = 1
                if not inst_mem_cls[j]:  # if inst mem is empty,
                    logger.debug('[%dth frame, %s(%d)] %dth box(LID:%d, %.3f) is added as a new inst in cinst_memory' % (
                    idx + delta, classes[j], j, i, new_inst.LID, new_inst.cls_score))
                    inst_mem_cls[j].append(new_inst)
                else:  # if local mem is not empty, comparte candidate box with local mem
                    for k, inst in enumerate(inst_mem_cls[j]):
                        coeff = compare_embed(inst.embed_feat, new_inst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                        coeff_th = (coeff > cfg.TEST.COEFF_THRESH_INTER)
                        iou = compute_IOU(inst.bbox,new_inst.bbox)
                        dist = compute_dist(inst, new_inst)
                        dist_th = (dist > cfg.TEST.DIST_THRESH_INTRA)
                        iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                        similar = iou * coeff
                        similar_th = (similar > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
                        logger.debug( '[%dth frame, %.5s(%d)] %dth box(LID:%d, %.3f) VS cinst_mem[%d](LID:%d, %.3f) (dis:%.3f(%.1s), IOU:%.3f(%.1s), coef:%.3f(%.1s), sim:%.3f(%.1s))' % (
                            idx + delta, classes[j], j, i, new_inst.LID, new_inst.cls_score, k, inst.LID, inst.cls_score, dist, dist_th, iou, iou_th, coeff, coeff_th, similar, similar_th))
                        if similar_th:  # when simillar
                            is_new -= 1
                            #inst.update_intra_frame(new_inst)
                            if new_inst.cls_score > inst.cls_score:
                                inst_mem_cls[j][k] = new_inst
                        elif not similar_th: # when not similar
                            continue
                        else:  # if sim is Nan or something
                            logger.error('[%dth frame, %.5s(%d)] error: sim value(%.3f) is not normal' % (idx + delta, classes[j], j, similar))
                            raise NotImplementedError

                    if int(is_new) is 1:
                        logger.debug('[%dth frame, %.5s(%d)] %dth box(LID:%d, %.3f) is added as a new inst in cinst_memory' % (
                            idx + delta, classes[j], j, i, new_inst.LID, new_inst.cls_score))
                        inst_mem_cls[j].append(new_inst)
                    else:
                        logger.debug('[%dth frame, %.5s(%d)] %dth box(LID:%d, %.3f) is not a new inst' % (
                            idx + delta, classes[j], j, i, new_inst.LID, new_inst.cls_score))

            inst_mem_now.extend(inst_mem_cls[j])

        sim_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final_th = np.zeros((1,len(inst_mem_now)), int)

        #logger.debug('[%dth frame] @@@ inter frame phase start @@@', idx + delta)
        if len(inst_mem) > 0 and len(inst_mem_now) > 0 :
            iou_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of iou
            coef_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of coeff
            sim_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of similarity
            g_rel_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of reliability
            l_rel_array = np.zeros((len(inst_mem), len(inst_mem_now))) # matrix of reliability g to l
            # Similarity matrix generation (global <-> local inst)
            for i, ginst in enumerate(inst_mem):  # loop for instances in this frame
                for j, linst in enumerate(inst_mem_now):
                    coeff = compare_embed(ginst.embed_feat, linst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                    iou = compute_IOU(ginst.bbox2, linst.bbox2) #use virtual bbox
                    similar = coeff * iou
                    g_reliability = similar * ginst.cls_score_reliable
                    l_reliability = similar * linst.cls_score * (ginst.cls_high == linst.cls)
                    coeff_th = (coeff > cfg.TEST.COEFF_THRESH_INTER)
                    iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                    similar_th = (similar > cfg.TEST.IOU_THRESH_INTRA * cfg.TEST.COEFF_THRESH_INTRA)
                    sim_array[i, j] = similar
                    iou_array[i, j] = iou
                    coef_array[i, j] = coeff
                    g_rel_array[i, j] = g_reliability
                    l_rel_array[i, j] = l_reliability
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %.5s(%.3f) lastf:%d)) VS linst[%d](LID:%d, %.5s(%.3f)): IOU:%.3f(%.1s) coef:%.3f(%.1s) sim:%.3f(%.1s) lrel:%.3f' % (
                        idx + delta, i, ginst.GID, classes[ginst.cls], ginst.cls_score, ginst.detected_idx[-1], j, linst.LID, classes[linst.cls], linst.cls_score,
                        iou, iou_th, coeff, coeff_th, similar, similar_th, l_reliability))

            sim_array_th = (sim_array[:, :] > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
            g_rel_array_th = (g_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            l_rel_array_th = (l_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            #sim_array_global.append(sim_array) #for debugging

            # ginst linking & ginst guided linst supression
            sim_array2 = sim_array.copy()
            sim_array2_th = sim_array_th.copy() * int(1) # bool to int
            g_rel_array2 = g_rel_array.copy()
            g_rel_array2_th = g_rel_array_th.copy()
            l_rel_array2 = l_rel_array.copy()
            link_array = np.zeros((len(inst_mem), len(inst_mem_now)))

            for i, ginst in enumerate(inst_mem):
                if ginst.g_suppressed:
                    continue
                lindex_over_th_sim = np.where(sim_array_th[i, :] == True)[0]
                lindex_over_th_lrel = np.where(l_rel_array_th[i, :] == True)[0]
                survive_idx = None
                gcls = ginst.cls_high
                same_cls_index_list = [] # find same class indexes
                same_cls_sim_list = [] #find same class sims
                same_cls_score_list = []  # find same class scores
                same_cls_grel_list = []  # find same class sims
                same_cls_lrel_list = []  # find same class sims
                for k, linst in enumerate(inst_mem_now):
                    if gcls == linst.cls:
                        same_cls_index_list.append(k)
                        same_cls_sim_list.append(sim_array[i, k])
                        same_cls_score_list.append(linst.cls_score)
                        same_cls_grel_list.append(g_rel_array[i, k])
                        same_cls_lrel_list.append(l_rel_array[i, k])

                if same_cls_index_list:
                    if len(lindex_over_th_lrel) > 0:
                        #survive_idx = same_cls_index_list[same_cls_sim_list.index(max(same_cls_sim_list))]
                        candidates = lindex_over_th_lrel
                        survive_idx = same_cls_index_list[same_cls_lrel_list.index(max(same_cls_lrel_list))]
                    elif len(lindex_over_th_sim) > 0:
                        candidates = lindex_over_th_sim
                        survive_idx = np.argmax(sim_array[i])

                if survive_idx is not None:
                    link_array[i, survive_idx] = 1
                    ginst.linked_to_LID = inst_mem_now[survive_idx].LID
                    lindex_to_suppress = np.setdiff1d(candidates[:], survive_idx)
                    sim_array2[:, lindex_to_suppress] = -1
                    sim_array2_th[:, lindex_to_suppress] = -1
                    g_rel_array2[:, lindex_to_suppress] = -1
                    g_rel_array2_th[:, lindex_to_suppress] = -1
                    for k in lindex_to_suppress:
                        inst_mem_now[k].l_suppressed.append(idx + delta)

            #linst guided ginst suppresion
            sim_array3 = sim_array2.copy()
            sim_array3_th = sim_array2_th.copy() * int(1) # bool to int
            rel_array3 = g_rel_array2.copy()
            rel_array3_th = g_rel_array2_th.copy()
            for i, linst in enumerate(inst_mem_now):
                if linst.l_suppressed:  # if not suppressed, suppress others
                    continue
                #gindex_over_th = np.where(sim_array2_th[:, i] == True)[0]
                gindex_over_th = np.where(g_rel_array2_th[:, i] == True)[0]
                if len(gindex_over_th) > 0:
                    survive = gindex_over_th.min() #oldest ginst survives
                    gindex_to_suppress = np.setdiff1d(gindex_over_th, survive) #others are suppressed
                    sim_array3[gindex_to_suppress, :] = -2
                    sim_array3_th[gindex_to_suppress, :] = -2
                    rel_array3[gindex_to_suppress, :] = -2
                    rel_array3_th[gindex_to_suppress, :] = -2
                    for k in gindex_to_suppress:
                        inst_mem[k].g_suppressed.append(idx + delta)

            sim_array_final = sim_array3.copy()
            sim_array_final_th = sim_array3_th.copy()
            rel_array_final = sim_array3.copy()
            rel_array_final_th = sim_array3_th.copy()
            linked_array = link_array.copy()

            # delete some ginsts
            for i, ginst in enumerate(inst_mem):
                if ginst.detected_idx[-1] < (idx + delta - cfg.TEST.GINST_LIFE_FRAME):
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (old frame)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    del inst_mem[i]
                    sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array = \
                        np.delete((sim_array_final,sim_array_final_th, rel_array_final, rel_array_final_th, linked_array),
                                  (i), axis=1)
                if ginst.g_suppressed:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (suppressed)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    del inst_mem[i] #TODO: elimination code must be written
                    sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array = \
                        np.delete((sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array),
                                  (i), axis=1)


            # link global insts & local insts
            #index_max_sim = np.zeros(len(inst_mem), dtype=int)
            #index_max_rel = np.zeros(len(inst_mem), dtype=int)
            for i, ginst in enumerate(inst_mem):
                #index_max_sim[i] = np.argmax(sim_array_final[i])
                #index_max_rel[i] = np.argmax(rel_array_final[i])
                indexes_linked = np.argwhere(linked_array[i] == 1)
                assert len(indexes_linked) >= 0
                if len(indexes_linked) == 0:
                    continue
                elif len(indexes_linked) > 1:
                    raise NotImplementedError # this should not happen
                index_linked = int(indexes_linked[0])
                linst = inst_mem_now[index_linked] ##TODO: change this
                max_sim = sim_array_final[i][index_linked]
                if max_sim >= (cfg.TEST.IOU_THRESH_INTER * cfg.TEST.COEFF_THRESH_INTER):
                    #inst_mem_now[sim_max_index[i]].linked_to.append(ginst.GID)
                    logger.debug(
                        '[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is updated by linst[%d](LID:%d, %s, %.3f)' % (
                            idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                            inst_mem[i].detected_idx[-1], index_linked, linst.LID, classes[linst.cls],
                            linst.cls_score))
                    ginst.update_inter_frame(linst, max_sim)
                else:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is not linked to anyone' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score, inst_mem[i].detected_idx[-1]))

            if idx == 39:  # 0.6
                logger.debug('debug_point')


        #make new ginst
        list_new_ginsts = []
        if len(inst_mem_now) > 0:
            for j, linst in enumerate(inst_mem_now):
                if linst.l_suppressed:
                    logger.debug(
                    '[%dth frame] linst[%d](LID:%d, %s, %.3f) is discarded because of suppression' % (
                        idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))
                    continue
                else:
                    if not len(linst.linked_to):  # for not linked one,
                        if len(inst_mem) > 0:
                            a = np.argmax(sim_array_final[:, j])
                            iou = iou_array[a, j]
                            if iou > 0.3 and inst_mem[a].detected_idx[-1] == (idx + delta): #if linst overlaps with ginst of this frame
                                logger.debug(
                                    '[%dth frame] linst[%d](LID:%d, %s, %.3f) is discarded because of overlap with ginst[%d](iou: %.3f)' % (
                                        idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, inst_mem[a].GID, iou))
                                continue

                        if linst.cls_score > cfg.TEST.SCORE_THRESH:
                            logger.debug('[%dth frame] linst[%d](LID:%d, %s, %.3f) became new ginst_memory[%d]' % (
                                idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, ginst_ID))
                            new_ginst = linst.make_global_inst(ginst_ID)
                            list_new_ginsts.append(new_ginst)
                            linst.linked_to = [ginst_ID]
                            ginst_ID = ginst_ID + 1

                        else:
                            logger.debug(
                                '[%dth frame] linst[%d](LID:%d, %s, %.3f) is discarded because of low cls_score' % (
                                    idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))

        inst_mem.extend(list_new_ginsts)

        #translate inst_mem_not into array
        for i, ginst in enumerate(inst_mem):
            if ginst.detected_idx[-1] == idx + delta:
                box_and_score = np.hstack((ginst.bbox, ginst.cls_score, ginst.cls_score_high))
                all_boxes_inst[ginst.cls_high][idx + delta].append(box_and_score)
        for i in range(num_classes):
            all_boxes_inst[i][idx + delta] = np.array(all_boxes_inst[i][idx + delta])

        out_im_ginst = 0
        out_im_linst = 0
        if vis:
            out_im_ginst = draw_all_ginst(center_image, inst_mem, classes, scales[delta], cfg, idx+delta)
            out_im_linst = draw_all_linst(center_image, inst_mem_now, inst_mem, classes, scales[delta], cfg)
            #out_im2 = draw_all_rois(center_image, rois, scales[delta], cfg) # print rois from RPN

        out_im = 0
        if cfg.TEST.SEQ_NMS == False and max_per_image > 0 and cfg.TEST.DISPLAY[0]:
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


    return ginst_ID, out_im, out_im_ginst, out_im_linst



