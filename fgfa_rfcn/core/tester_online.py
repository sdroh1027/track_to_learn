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
    pred_boxes_on_feat[:, 0] = pred_boxes[:, 0] * scale_w  # x1
    pred_boxes_on_feat[:, 1] = pred_boxes[:, 1] * scale_h  # y1
    pred_boxes_on_feat[:, 2] = pred_boxes[:, 2] * scale_w  # x2
    pred_boxes_on_feat[:, 3] = pred_boxes[:, 3] * scale_h  # y2
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

    return cropped_embed[0]

def roi_crop_embed_nd(img_height, img_width, pred_boxes, cur_embed, cfg):
    # using cur_embed & bbox_pred, make new instance wise embed layer
    # bbox_pred[4:8] has (x1,y1) (x2,y2) coordinates

    scale_h = cur_embed.shape[2] / img_height
    scale_w = cur_embed.shape[3] / img_width

    zeros = np.zeros([300, 1])
    pred_boxes_on_feat = np.zeros([300, 4])
    pred_boxes_on_feat[:, 0] = pred_boxes[:, 0] * scale_w  # x1
    pred_boxes_on_feat[:, 1] = pred_boxes[:, 1] * scale_h  # y1
    pred_boxes_on_feat[:, 2] = pred_boxes[:, 2] * scale_w  # x2
    pred_boxes_on_feat[:, 3] = pred_boxes[:, 3] * scale_h  # y2
    pred_boxes_on_feat = clip_boxes(pred_boxes_on_feat, cur_embed.shape[-2:])
    pred_boxes_on_feat = np.hstack((zeros, pred_boxes_on_feat[:, 0:]))
    pred_boxes_on_feat = mx.nd.array(pred_boxes_on_feat, mx.gpu())

    new_embed = mx.nd.contrib.ROIAlign(data=cur_embed, rois=pred_boxes_on_feat,
                                  pooled_size=(cfg.TEST.EMBED_SIZE, cfg.TEST.EMBED_SIZE),
                                  spatial_scale=1.0, name='new_embed')
    new_embed_normalized = mx.nd.L2Normalization(data=new_embed, mode='channel', name='new_embed_normalized')

    return new_embed_normalized

def im_detect_all(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)
    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    iscores_all = [] # instance prediction
    ipred_boxes_all = [] # instance prediction
    rois_all = []
    cropped_embeds_all = []
    psroipooled_cls_rois_all = []

    # for debugging
    # cur_embeds_all = [] #
    # new_embeds_all = [] #
    # sliced_all = [] #

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
        psroipooled_cls_rois_nd = output['psroipooled_cls_rois_output']
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
        cropped_embed = roi_crop_embed_nd(img_height, img_width, rois, cur_embed, cfg) #nd_array is faster
        cropped_embed = cropped_embed.asnumpy()

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
        psroipooled_cls_rois_all.append(psroipooled_cls_rois_nd)

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
                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s GID:{} #GInsts:{}'.format(idx, num_images,
                                                                                      data_time / idx * test_data.batch_size,
                                                                                      net_time / idx * test_data.batch_size,
                                                                                      post_time / idx * test_data.batch_size, ginst_ID, len(ginst_mem))
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s GID:{} #GInsts:{}'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size, ginst_ID, len(ginst_mem)))
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

                print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s GID:{} #GInsts:{}'.format(idx, num_images,
                                                                                   data_time / idx * test_data.batch_size,
                                                                                   net_time / idx * test_data.batch_size,
                                                                                   post_time / idx * test_data.batch_size, ginst_ID, len(ginst_mem))
                if logger:
                    logger.info('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s GID:{} #GInsts:{}'.format(idx, num_images,
                                                                                             data_time / idx * test_data.batch_size,
                                                                                             net_time / idx * test_data.batch_size,
                                                                                             post_time / idx * test_data.batch_size, ginst_ID, len(ginst_mem)))
                end_counter += 1

    with open(det_file, 'wb') as f:
        cPickle.dump((all_boxes_inst, frame_ids), f, protocol=cPickle.HIGHEST_PROTOCOL)

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
    info_str = imdb.evaluate_detections_multiprocess_select_idx(res, -5)
    info_str2 = imdb.evaluate_detections_multiprocess_select_idx(res, -4)
    info_str3 = imdb.evaluate_detections_multiprocess_select_idx(res, -3)
    info_str4 = imdb.evaluate_detections_multiprocess_select_idx(res, -2)
    info_str5 = imdb.evaluate_detections_multiprocess_select_idx(res, -1)

    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))
        logger.info('evaluate detections: \n{}'.format(info_str2))
        logger.info('evaluate detections: \n{}'.format(info_str3))
        logger.info('evaluate detections: \n{}'.format(info_str4))
        logger.info('evaluate detections: \n{}'.format(info_str5))

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
            cv2.putText(im, 'LID:%d %.3s %.2f GID:%s %.3s s:%.2f' % (inst.LID, class_names[cls], score, inst.GID,
                        class_names[inst.cls_high], inst.sim),
                        (bbox[0], bbox[1] + 10), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def draw_all_linst(im_array, inst_mem_now, inst_mem, class_names, scale, cfg, threshold=0.01): # sidong
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
        if inst.cls_score < threshold:
            continue
        if inst.linked_to_gidx >= 0:
            inst = inst_mem[inst.linked_to_gidx]
            if not inst.color:
                inst.color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
            bbox = inst.bbox[:4] * scale
            cls = inst.cls
            score = inst.cls_score

            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=inst.color, thickness=2)
            cv2.putText(im, 'ID:%d %.3s %.2f GID:%s %.3s s:%.2f' % (inst.LID, class_names[cls], score, inst.GID,
                        class_names[inst.cls_high], inst.sim),
                        (bbox[0], bbox[1] + 10), color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        else:
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
    return sum.sum()/sum.size

def normalize(ndarray):
    return ndarray/(ndarray.square().sum().sqrt())

def compare_embed_filtered(A, B, normed_score_map_A, normed_score_map_B):
    # score_map : mxnet nd array (7*7)
    # output: cosine similarity

    C = A * B # C vector: <type 'tuple'>: (2048, z,z) (z*z vectors, length: 2048)
    cos_sim = np.sum(C, axis=0)  # return vector: <type 'tuple'>: (z,z)

    ggrid = normed_score_map_A * normed_score_map_B # (7, 7) grid matrix
    grid1 = mx.nd.softmax(data=ggrid.reshape(-3), axis=-1, temperature=0.05).reshape((7,7)).asnumpy()
    filtered_sim1 = grid1 * cos_sim
    sum1 = filtered_sim1.sum()
    #grid11 = mx.nd.softmax(data=ggrid.reshape(-3), axis=-1, temperature=0.1).reshape((7, 7)).asnumpy()

    #soft_A = mx.nd.softmax(normed_score_map_A.reshape(-3), axis=-1, temperature=0.1)
    #aa = soft_A.asnumpy().reshape((7,7))
    #soft_B = mx.nd.softmax(normed_score_map_B.reshape(-3), axis=-1, temperature=0.1)
    #bb = soft_B.asnumpy().reshape((7,7))
    #grid2 = mx.nd.softmax((soft_A * soft_B), axis=-1, temperature=0.01).reshape(7,7).asnumpy()
    #filtered_sim2 = grid2 * cos_sim
    #sum2 = filtered_sim2.sum()

    #AAA = score_map_A.asnumpy()
    #BBB = score_map_B.asnumpy()
    #DDD = ggrid.asnumpy()

    return sum1

def Area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def compute_IOU(box_A,box_B, area_A, area_B):
    xx1 = max(box_A[0],box_B[0])
    yy1 = max(box_A[1],box_B[1])
    xx2 = min(box_A[2],box_B[2])
    yy2 = min(box_A[3],box_B[3])

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (area_A + area_B - inter)
    return ovr

def compute_IOM(box_me,box_B):
    area_A = Area(box_me)

    xx1 = max(box_me[0], box_B[0])
    yy1 = max(box_me[1], box_B[1])
    xx2 = min(box_me[2], box_B[2])
    yy2 = min(box_me[3], box_B[3])

    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (area_A)
    return ovr

def compute_relative_dist(inst_A, inst_B):
    rad_A = math.sqrt(Area(inst_B.bbox))
    rad_B = math.sqrt(Area(inst_B.bbox))
    dx = inst_A.center[0] - inst_B.center[0]
    dy = inst_A.center[1] - inst_B.center[1]
    dis = math.sqrt(dx ** 2 + dy ** 2)
    return ((rad_A + rad_B) - dis)/(rad_A + rad_B)

def compute_locality(inst_A, inst_B):
    rd = compute_relative_dist(inst_A, inst_B)
    return (0 if rd <= 0 else rd)

def ginst_to_box_and_score(inst):
    dynamic_average = inst.cls_score_acc / len(inst.detected_idx)
    box_and_score = np.hstack(
        (inst.bbox, inst.cls_score, inst.cls_score_high, inst.cls_score_reliable,
         dynamic_average, max(dynamic_average, inst.cls_score)
         ))
    return box_and_score

def ginst_to_box_and_score_cls_scores(inst):
    dynamic_average = inst.cls_score_acc / len(inst.detected_idx)
    box_and_score = np.hstack(
        (inst.bbox, inst.cls_score, inst.cls_score_high, inst.cls_score_reliable,
         dynamic_average, max(dynamic_average, inst.cls_score)
         ))
    return box_and_score

def linst_to_box_and_score(inst):
    box_and_score = np.hstack(
        (inst.bbox, inst.cls_score, inst.cls_score_high, inst.cls_score,
         inst.cls_score, inst.cls_score))
    return box_and_score

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
        all_cls_box_idx_nms = []
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
                #cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]
                per_cls_box_idx_nms.append(indexes[keep])
                #all_cls_box_idx_nms.extend(indexes[keep])

        #inst_mem_cls = [[] for _ in range(num_classes)]  ## intra-class inst memory
        #for j in range(1, num_classes):
            box_indexes = per_cls_box_idx_nms[j]
            for i, box_idx in enumerate(box_indexes): # for dets of class j,
                box = all_boxes[j][idx + delta][i]
                new_inst = Instance(LID=local_inst_ID, frame_idx=idx + delta, cls=j, cls_score=box[-1],
                                    bbox=box[0:4],  embed_feat=embed_feat[box_idx],
                                    atten=normalize(psroipooled_cls_rois[box_idx][j])) #make instances for outputs
                new_inst.cls_scores = box_idx
                local_inst_ID += 1
                #inst_mem_cls[j].append(new_inst)
                inst_mem_now.append(new_inst)
            #inst_mem_now.extend(inst_mem_cls[j])

        iou_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final_th = np.zeros((1,len(inst_mem_now)), int)
        linked_array = np.zeros((len(inst_mem), len(inst_mem_now)))
        gidx_linked = []
        #logger.debug('[%dth frame] @@@ inter frame phase start @@@', idx + delta)
        len_local = len(inst_mem_now)
        len_global = len(inst_mem)
        if len_local > 0 and len_global > 0 :
            iou_array = np.zeros((len_global, len_local))  # matrix of iou
            coef_array = np.zeros((len_global, len_local))  # matrix of coeff
            sim_array = np.zeros((len_global, len_local))  # matrix of similarity
            g_rel_array = np.zeros((len_global, len_local))  # matrix of reliability
            same_cls_score_array = np.zeros((len_global, len_local))  # matrix of reliability
            l_rel_array = np.zeros((len_global, len_local)) # matrix of reliability g to l
            #loc_array = np.zeros((len_global, len_local)) # matrix of relative distance

            area1 = np.empty(len_global)
            for i, ginst in enumerate(inst_mem):
                area1[i] = Area(ginst.bbox2)
            area2 = np.empty(len_local)
            for j, linst in enumerate(inst_mem_now):
                area2[j] = Area(linst.bbox2)

            # Similarity matrix generation (global <-> local inst)
            for i, ginst in enumerate(inst_mem):  # loop for instances in this frame
                for j, linst in enumerate(inst_mem_now):
                    coeff = 0 #compare_embed(ginst.embed_feat, linst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                    coeff_filtered = compare_embed_filtered(ginst.embed_feat, linst.embed_feat, ginst.atten, linst.atten)
                    iou = compute_IOU(ginst.bbox2, linst.bbox2, area1[i], area2[j]) #use virtual bbox
                    similar = coeff_filtered * iou
                    #g_reliability = similar * ginst.cls_score_reliable
                    same_cls_score = linst.cls_score * (ginst.cls_high == linst.cls)
                    l_reliability = similar * same_cls_score
                    #loc = compute_locality(ginst, linst)
                    coeff_th = (coeff > cfg.TEST.COEFF_THRESH_INTER)
                    iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                    similar_th = (similar > cfg.TEST.IOU_THRESH_INTRA * cfg.TEST.COEFF_THRESH_INTRA)
                    sim_array[i, j] = similar
                    iou_array[i, j] = iou
                    coef_array[i, j] = coeff_filtered #
                    #g_rel_array[i, j] = g_reliability
                    same_cls_score_array[i, j] = same_cls_score
                    l_rel_array[i, j] = l_reliability
                    #loc_array[i, j] = loc
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %.5s(%.2f) lastf:%d)) VS linst[%d](LID:%d, %.5s(%.2f)): filt_coef:%.2f IOU:%.2f(%.1s) coef:%.2f(%.1s) sim:%.2f(%.1s) lrel:%.2f' % (
                        idx + delta, i, ginst.GID, classes[ginst.cls], ginst.cls_score, ginst.detected_idx[-1], j, linst.LID, classes[linst.cls], linst.cls_score,
                        coeff_filtered, iou, iou_th, coeff, coeff_th, similar, similar_th, l_reliability))

            sim_array_th = (sim_array[:, :] > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
            g_rel_array_th = (g_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            l_rel_array_th = (l_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            #sim_array_global.append(sim_array) #for debugging

            link_array = np.zeros((len_global, len_local))
            sim_array_tmp = sim_array.copy()
            l_rel_array_tmp = l_rel_array.copy()
            sim_array_hor = sim_array.copy()
            l_rel_array_hor = l_rel_array.copy()
            sim_array_ver = sim_array.copy()
            l_rel_array_ver = l_rel_array.copy()
            same_cls_score_array_tmp = same_cls_score_array.copy()

            trial = 0
            linked_num = 0
            # ginst linking
            while l_rel_array_tmp.max() > cfg.TEST.REL_THRESH_INTRA and linked_num < min(15, len_global):
                #i = l_rel_array_tmp.argmax()
                i = same_cls_score_array_tmp.argmax()
                gi = i / len_local
                li = i % len_local
                trial += 1
                if l_rel_array_tmp[gi, li] > cfg.TEST.REL_THRESH_INTRA:
                    link_array[gi, li] = 1
                    sim_array_tmp[gi, :] = -1
                    sim_array_tmp[:, li] = -1
                    l_rel_array_tmp[gi, :] = -1
                    l_rel_array_tmp[:, li] = -1
                    sim_array_hor[gi, :] = -1
                    sim_array_ver[:, li] = -1
                    l_rel_array_hor[gi, :] = -1
                    l_rel_array_ver[:, li] = -1
                    same_cls_score_array_tmp[gi, :] = -1
                    same_cls_score_array_tmp[:, li] = -1
                    gidx_linked.append(gi)
                    inst_mem[gi].update_inter_frame(inst_mem_now[li], sim_array[gi,li])
                    linked_num += 1
                else:
                    sim_array_tmp[gi, li] = -1
                    l_rel_array_tmp[gi, li] = -1
                    sim_array_hor[gi, li] = -1
                    sim_array_ver[gi, li] = -1
                    l_rel_array_hor[gi, li] = -1
                    l_rel_array_ver[gi, li] = -1
                    same_cls_score_array_tmp[gi, li] = -1


            while 0:#sim_array_tmp.max() > cfg.TEST.IOU_THRESH_INTER * cfg.TEST.COEFF_THRESH_INTER:
                i = sim_array_tmp.argmax()
                gi = i / len(inst_mem_now)
                li = i % len(inst_mem_now)
                link_array[gi, li] = 2
                sim_array_tmp[gi, :] = -2
                sim_array_tmp[:, li] = -2
                l_rel_array_tmp[gi, :] = -2
                l_rel_array_tmp[:, li] = -2
                sim_array_hor[gi, :] = -2
                sim_array_ver[:, li] = -2
                l_rel_array_hor[gi, :] = -2
                l_rel_array_ver[:, li] = -2
                gidx_linked.append(gi)
                inst_mem[gi].update_inter_frame(inst_mem_now[li], sim_array[gi, li])

            l_rel_array_ver_th = (l_rel_array_ver[:, :] > cfg.TEST.REL_THRESH_INTRA)
            l_rel_array_hor_th = (l_rel_array_hor[:, :] > cfg.TEST.REL_THRESH_INTRA)

            # ginst guided linst supression (for not linked)
            if cfg.TEST.LSUP:
                for i, ginst in enumerate(inst_mem):
                    if link_array[i, :].max() > 0:
                        lindex_to_suppress = np.where(l_rel_array_ver_th[i, :] == True)[0]
                        sim_array_tmp[:, lindex_to_suppress] = -3
                        for k in lindex_to_suppress:
                            inst_mem_now[k].l_suppressed.append(idx + delta)
            # linst guided ginst suppresion (for not linked)
            if cfg.TEST.GSUP:
                for j, linst in enumerate(inst_mem_now):
                    if link_array[:, j].max() > 0:
                        gindex_to_suppress = np.where(l_rel_array_hor_th[:, j] == True)[0]
                        sim_array_tmp[gindex_to_suppress, :] = -4
                        for k in gindex_to_suppress:
                            inst_mem[k].g_suppressed.append(idx + delta)

            iou_array_final = iou_array.copy()
            sim_array_final = sim_array.copy()
            sim_array_final_th = sim_array_th.copy()
            rel_array_final = l_rel_array.copy()
            rel_array_final_th = l_rel_array_th.copy()
            linked_array = link_array.copy()

            # delete some ginsts
            gidx_to_delete = []
            for i, ginst in enumerate(inst_mem):
                delete = 0
                last_frame = ginst.detected_idx[-1]
                if last_frame < (idx + delta - cfg.TEST.GINST_LIFE_FRAME):
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (old frame)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                elif ginst.g_suppressed:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (suppressed)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                elif 0:#ginst.cls_score_reliable < 0.05:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (low reliability)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                if delete == 1:
                    gidx_to_delete.append(i)
                    if last_frame == idx + delta:
                        gidx_linked.remove(i)

            gidx_to_delete.reverse()
            for i in gidx_to_delete:
                del inst_mem[i]
            iou_array_final, sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array = \
                np.delete(
                    (iou_array_final, sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array),
                    (gidx_to_delete), axis=1)


            # link global insts & local insts
            for i, ginst in enumerate(inst_mem):
                indexes_linked = np.argwhere(linked_array[i, :] >= 1)
                if len(indexes_linked) == 0:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is not linked to anyone' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                elif len(indexes_linked) == 1:
                    index_linked = int(indexes_linked[0])
                    linst = inst_mem_now[index_linked]
                    sim = sim_array_final[i][index_linked]
                    logger.debug(
                        '[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is updated by linst[%d](LID:%d, %s, %.3f)' % (
                            idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                            inst_mem[i].detected_idx[-1], index_linked, linst.LID, classes[linst.cls],
                            linst.cls_score))
                    #ginst.update_inter_frame(linst, sim)
                else:
                    logger.debug('# of linked indexes %d' % (len(indexes_linked)))
                    raise NotImplementedError # this should not happen

            #if idx == 88:  # 0.6
            #    logger.debug('debug_point')

        # make new ginst
        # translate inst_mem_now into box array (for evaluation)
        gidx_linked_from_lidx = [None for _ in range(len(inst_mem_now))]
        list_new_ginsts = []
        if len(inst_mem_now) > 0:
            for j, linst in enumerate(inst_mem_now):
                if linst.l_suppressed: # if suppressed
                    logger.debug(
                        '[%dth frame] linst[%d](LID:%d, %s, %.3f) is discarded because of suppression' % (
                            idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))
                    continue
                gidx_linked_from_lidx[j] = np.argwhere(linked_array[:, j] >= 1)
                if len(gidx_linked_from_lidx[j]): # if linked to ginst
                    assert len(gidx_linked_from_lidx[j]) == 1
                    gidx = gidx_linked_from_lidx[j][0][0]
                    ginst = inst_mem[gidx]
                    box_and_score = ginst_to_box_and_score(ginst)
                    linst.cls_high = ginst.cls_high
                    linst.linked_to_gidx = gidx

                else: # if not linked
                    if linst.cls_score >= cfg.TEST.SCORE_THRESH:
                        if len(inst_mem) > 0:
                            iou_sum = iou_array[gidx_linked, j].sum()
                            #compute_IOM(box_me=, box_B=)
                            # exception handling
                            if iou_sum > 0.5:  # if linst overlaps with linked ginsts of this frame
                                iou_max_index = np.argmax(iou_array_final[:, j])
                                if linst.cls == inst_mem[iou_max_index].cls and linst.cls_score < inst_mem[iou_max_index].cls_score:
                                    logger.debug(
                                        '[%dth frame] linst[%d](LID:%d, %s, %.3f) will be remained linst because of overlap(%.2f) with ginsts{g_linked}'.format(
                                            g_linked=gidx_linked)
                                        % (idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, iou_sum))
                                    box_and_score = linst_to_box_and_score(linst)
                                    continue
                        logger.debug('[%dth frame] linst[%d](LID:%d, %s, %.3f) became new ginst_memory[%d]' % (
                            idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, ginst_ID))

                        new_ginst = linst.make_global_inst(ginst_ID)
                        list_new_ginsts.append(new_ginst)
                        linst.linked_to = [ginst_ID]
                        linst.linked_to_gidx = len(inst_mem) + len(list_new_ginsts) - 1
                        ginst_ID = ginst_ID + 1
                        box_and_score = ginst_to_box_and_score(new_ginst)
                    else:
                        logger.debug(
                            '[%dth frame] linst[%d](LID:%d, %s, %.3f) will be remained linst because of low cls_score' % (
                                idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))
                        box_and_score = linst_to_box_and_score(linst)

                all_boxes_inst[linst.cls_high][idx + delta].append(box_and_score)
            for i in range(num_classes):
                all_boxes_inst[i][idx + delta] = np.array(all_boxes_inst[i][idx + delta])
        inst_mem.extend(list_new_ginsts)

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


def process_link_pred_result2(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, all_boxes_inst, idx,
                             max_per_image, vis, center_image, scales, inst_mem, sim_array_global, ginst_ID):
    global logger

    for delta, (scores, boxes, rois, data_dict, iscores, ipred_boxes, embed_feat, psroipooled_cls_rois) in enumerate(pred_result): #16th frame -> cat(10) -> local box(3, 4th in rois)
        local_inst_ID = 0
        inst_mem_now = [] # intra-frame inst memory

        per_cls_box_idx_over_th = [[]]
        per_cls_box_idx_nms = [[]]
        all_cls_box_idx_nms = []
        for j in range(1, num_classes):
            # logger.debug('[%dth frame]test boxes of class: (%d, %s)' % (idx+delta, j, classes[j]))
            indexes = np.where(scores[:, j] > thresh)[0]
            per_cls_box_idx_over_th.append(indexes)
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][
                    idx + delta] = cls_dets  # all_boxes[31L(class)][frame_idx + batch_size][300L(box_idx)][5L(x,y,w,h,score)]
            else:
                # cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]
                per_cls_box_idx_nms.append(indexes[keep])
                # all_cls_box_idx_nms.extend(indexes[keep])

        #merging



        scores_max = scores[:,1:].max(axis=1)
        index_tf = (scores_max > thresh)
        cls_scores = scores[index_tf]
        cls_highest = scores[index_tf, 1:].argmax(axis=1) + 1
        cls_boxes = boxes[index_tf, 4:8] if cfg.CLASS_AGNOSTIC else boxes[index_tf, j * 4:(j + 1) * 4]
        embed_feats = embed_feat[index_tf]
        psroipooled_cls_roiss = psroipooled_cls_rois[index_tf]

        for i in range(len(cls_scores)):
            new_inst = Instance(LID=local_inst_ID, frame_idx=idx + delta, cls=cls_highest[i], cls_score=cls_scores[i][cls_highest[i]],
                                bbox=cls_boxes[i], embed_feat=embed_feats[i],
                                atten=psroipooled_cls_roiss[i])  # make instances for outputs
            new_inst.cls_scores = cls_scores[i]
            new_inst.cls_scores_acc = cls_scores[i]
            local_inst_ID += 1
            inst_mem_now.append(new_inst)

        logger.debug('[%dth frame] %d linsts created' % (idx + delta, len(inst_mem_now)))

        iou_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final = np.zeros((1,len(inst_mem_now)))
        sim_array_final_th = np.zeros((1,len(inst_mem_now)), int)
        linked_array = np.zeros((len(inst_mem), len(inst_mem_now)))
        gidx_linked = []
        #logger.debug('[%dth frame] @@@ inter frame phase start @@@', idx + delta)
        if len(inst_mem) > 0 and len(inst_mem_now) > 0 :
            cls_sim_array = np.zeros((len(inst_mem), len(inst_mem_now)))
            iou_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of iou
            coef_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of coeff
            sim_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of similarity
            g_rel_array = np.zeros((len(inst_mem), len(inst_mem_now)))  # matrix of reliability
            l_rel_array = np.zeros((len(inst_mem), len(inst_mem_now))) # matrix of reliability g to l
            loc_array = np.zeros((len(inst_mem), len(inst_mem_now))) # matrix of relative distance

            area1 = np.empty(len(inst_mem))
            for i, ginst in enumerate(inst_mem):
                area1[i] = Area(ginst.bbox2)
            area2 = np.empty(len(inst_mem_now))
            for j, linst in enumerate(inst_mem_now):
                area2[j] = Area(linst.bbox2)

            # Similarity matrix generation (global <-> local inst)
            for i, ginst in enumerate(inst_mem):  # loop for instances in this frame
                for j, linst in enumerate(inst_mem_now):
                    cls_sim = (linst.cls_scores * ginst.cls_scores).sum()
                    coeff = 0 #compare_embed(ginst.embed_feat, linst.embed_feat)  # input embed_feat <type 'tuple'>: (2048, 8,8)
                    iou = compute_IOU(ginst.bbox2, linst.bbox2, area1[i], area2[j]) #use virtual bbox
                    similar = iou * cls_sim #original ver
                    g_reliability = 0 # similar * ginst.cls_score_reliable
                    l_reliability = 0 # similar * linst.cls_scores[ginst.cls]
                    loc = 0 # compute_locality(ginst, linst)
                    coeff_th = 0 # (coeff > cfg.TEST.COEFF_THRESH_INTER)
                    iou_th = (iou > cfg.TEST.IOU_THRESH_INTRA)
                    similar_th = (similar > cfg.TEST.IOU_THRESH_INTRA * cfg.TEST.COEFF_THRESH_INTRA)

                    cls_sim_array[i, j] = cls_sim
                    iou_array[i, j] = iou
                    sim_array[i, j] = similar
                    #coef_array[i, j] = coeff
                    #g_rel_array[i, j] = g_reliability
                    #l_rel_array[i, j] = l_reliability
                    #loc_array[i, j] = loc
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %.5s(%.2f) lastf:%d)) VS linst[%d](LID:%d, %.5s(%.2f)): cls_sim:%.2f IOU:%.2f(%.1s) coef:%.2f(%.1s) sim:%.2f(%.1s) lrel:%.2f' % (
                        idx + delta, i, ginst.GID, classes[ginst.cls], ginst.cls_score, ginst.detected_idx[-1], j, linst.LID, classes[linst.cls], linst.cls_score,
                        cls_sim, iou, iou_th, coeff, coeff_th, similar, similar_th, l_reliability))

            sim_array_th = (sim_array[:, :] > cfg.TEST.COEFF_THRESH_INTRA * cfg.TEST.IOU_THRESH_INTRA)
            g_rel_array_th = (g_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            l_rel_array_th = (l_rel_array[:, :] > cfg.TEST.REL_THRESH_INTRA)
            #sim_array_global.append(sim_array) #for debugging

            link_array = np.zeros((len(inst_mem), len(inst_mem_now)))
            sim_array_tmp = sim_array.copy()
            l_rel_array_tmp = l_rel_array.copy()
            sim_array_hor = sim_array.copy()
            l_rel_array_hor = l_rel_array.copy()
            sim_array_ver = sim_array.copy()
            l_rel_array_ver = l_rel_array.copy()

            # ginst linking
            while 0: # l_rel_array_tmp.max() > cfg.TEST.REL_THRESH_INTRA:
                i = l_rel_array_tmp.argmax()
                gi = i / len(inst_mem_now)
                li = i % len(inst_mem_now)
                link_array[gi, li] = 1
                sim_array_tmp[gi, :] = -1
                sim_array_tmp[:, li] = -1
                l_rel_array_tmp[gi, :] = -1
                l_rel_array_tmp[:, li] = -1
                sim_array_hor[gi, :] = -1
                sim_array_ver[:, li] = -1
                l_rel_array_hor[gi, :] = -1
                l_rel_array_ver[:, li] = -1
                gidx_linked.append(gi)
                inst_mem[gi].update_inter_frame(inst_mem_now[li], sim_array[gi,li])

            while sim_array_tmp.max() > 0: # cfg.TEST.IOU_THRESH_INTER * cfg.TEST.COEFF_THRESH_INTER:
                i = sim_array_tmp.argmax()
                gi = i / len(inst_mem_now)
                li = i % len(inst_mem_now)
                link_array[gi, li] = 2
                sim_array_tmp[gi, :] = -2
                sim_array_tmp[:, li] = -2
                l_rel_array_tmp[gi, :] = -2
                l_rel_array_tmp[:, li] = -2
                sim_array_hor[gi, :] = -2
                sim_array_ver[:, li] = -2
                l_rel_array_hor[gi, :] = -2
                l_rel_array_ver[:, li] = -2
                gidx_linked.append(gi)
                inst_mem[gi].update_inter_frame(inst_mem_now[li], sim_array[gi, li])
                inst_mem[gi].cls_scores_acc += inst_mem_now[li].cls_scores

            l_rel_array_ver_th = (l_rel_array_ver[:, :] > cfg.TEST.REL_THRESH_INTRA)
            l_rel_array_hor_th = (l_rel_array_hor[:, :] > cfg.TEST.REL_THRESH_INTRA)

            # ginst guided linst supression (for not linked)
            if cfg.TEST.LSUP:
                for i, ginst in enumerate(inst_mem):
                    if link_array[i, :].max() > 0:
                        lindex_to_suppress = np.where(l_rel_array_ver_th[i, :] == True)[0]
                        sim_array_tmp[:, lindex_to_suppress] = -3
                        for k in lindex_to_suppress:
                            inst_mem_now[k].l_suppressed.append(idx + delta)
            # linst guided ginst suppresion (for not linked)
            if cfg.TEST.GSUP:
                for j, linst in enumerate(inst_mem_now):
                    if link_array[:, j].max() > 0:
                        gindex_to_suppress = np.where(l_rel_array_hor_th[:, j] == True)[0]
                        sim_array_tmp[gindex_to_suppress, :] = -4
                        for k in gindex_to_suppress:
                            inst_mem[k].g_suppressed.append(idx + delta)

            iou_array_final = iou_array.copy()
            sim_array_final = sim_array.copy()
            sim_array_final_th = sim_array_th.copy()
            rel_array_final = l_rel_array.copy()
            rel_array_final_th = l_rel_array_th.copy()
            linked_array = link_array.copy()

            # delete some ginsts
            gidx_to_delete = []
            for i, ginst in enumerate(inst_mem):
                delete = 0
                last_frame = ginst.detected_idx[-1]
                if last_frame < (idx + delta - cfg.TEST.GINST_LIFE_FRAME):
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (old frame)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                elif ginst.g_suppressed:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (suppressed)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                elif 0:#ginst.cls_score_reliable < 0.05:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is deleted (low reliability)' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                    delete = 1
                if delete == 1:
                    gidx_to_delete.append(i)
                    if last_frame == idx + delta:
                        gidx_linked.remove(i)

            gidx_to_delete.reverse()
            for i in gidx_to_delete:
                del inst_mem[i]
            iou_array_final, sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array = \
                np.delete(
                    (iou_array_final, sim_array_final, sim_array_final_th, rel_array_final, rel_array_final_th, linked_array),
                    (gidx_to_delete), axis=1)


            # link global insts & local insts
            for i, ginst in enumerate(inst_mem):
                indexes_linked = np.argwhere(linked_array[i, :] >= 1)
                if len(indexes_linked) == 0:
                    logger.debug('[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is not linked to anyone' % (
                        idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                        inst_mem[i].detected_idx[-1]))
                elif len(indexes_linked) == 1:
                    index_linked = int(indexes_linked[0])
                    linst = inst_mem_now[index_linked]
                    sim = sim_array_final[i][index_linked]
                    logger.debug(
                        '[%dth frame] ginst[%d](GID:%d, %s, %.3f, frame:%d) is updated by linst[%d](LID:%d, %s, %.3f)' % (
                            idx + delta, i, inst_mem[i].GID, classes[inst_mem[i].cls], inst_mem[i].cls_score,
                            inst_mem[i].detected_idx[-1], index_linked, linst.LID, classes[linst.cls],
                            linst.cls_score))
                    #ginst.update_inter_frame(linst, sim)
                else:
                    logger.debug('# of linked indexes %d' % (len(indexes_linked)))
                    raise NotImplementedError # this should not happen

            #if idx == 88:  # 0.6
            #    logger.debug('debug_point')

        # make new ginst
        # translate inst_mem_now into box array (for evaluation)
        gidx_linked_from_lidx = [None for _ in range(len(inst_mem_now))]
        list_new_ginsts = []
        if len(inst_mem_now) > 0:
            for j, linst in enumerate(inst_mem_now):
                if linst.l_suppressed: # if suppressed
                    logger.debug(
                        '[%dth frame] linst[%d](LID:%d, %s, %.3f) is discarded because of suppression' % (
                            idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))
                    continue
                gidx_linked_from_lidx[j] = np.argwhere(linked_array[:, j] >= 1)
                if len(gidx_linked_from_lidx[j]): # if linked to ginst
                    assert len(gidx_linked_from_lidx[j]) == 1
                    gidx = gidx_linked_from_lidx[j][0][0]
                    ginst = inst_mem[gidx]
                    box_and_score = ginst_to_box_and_score_cls_scores(ginst)
                    linst.cls_high = ginst.cls_scores_acc[1:].argmax() + 1 # this cls result is only accurate for dynamic averaging result
                    linst.linked_to_gidx = gidx

                else: # if not linked
                    if linst.cls_score >= cfg.TEST.SCORE_THRESH:
                        if len(inst_mem) > 0:
                            iou_sum = iou_array[gidx_linked, j].sum()
                            #compute_IOM(box_me=, box_B=)
                            # exception handling
                            if iou_sum > 0.5:  # if linst overlaps with linked ginsts of this frame
                                iou_max_index = np.argmax(iou_array_final[:, j])
                                if linst.cls == inst_mem[iou_max_index].cls and linst.cls_score < inst_mem[iou_max_index].cls_score:
                                    logger.debug(
                                        '[%dth frame] linst[%d](LID:%d, %s, %.3f) will be remained linst because of overlap(%.2f) with ginsts{g_linked}'.format(
                                            g_linked=gidx_linked)
                                        % (idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, iou_sum))
                                    box_and_score = linst_to_box_and_score(linst)
                                    continue
                        logger.debug('[%dth frame] linst[%d](LID:%d, %s, %.3f) became new ginst_memory[%d]' % (
                            idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score, ginst_ID))

                        new_ginst = linst.make_global_inst(ginst_ID)
                        new_ginst.cls_scores = linst.cls_scores
                        new_ginst.cls_scores_acc = linst.cls_scores_acc
                        list_new_ginsts.append(new_ginst)
                        linst.linked_to = [ginst_ID]
                        linst.linked_to_gidx = len(inst_mem) + len(list_new_ginsts) - 1
                        ginst_ID = ginst_ID + 1
                        box_and_score = ginst_to_box_and_score_cls_scores(new_ginst)
                    else:
                        logger.debug(
                            '[%dth frame] linst[%d](LID:%d, %s, %.3f) will be remained linst because of low cls_score' % (
                                idx + delta, j, linst.LID, classes[linst.cls], linst.cls_score))
                        box_and_score = linst_to_box_and_score(linst)

                all_boxes_inst[linst.cls_high][idx + delta].append(box_and_score)
            for i in range(num_classes):
                all_boxes_inst[i][idx + delta] = np.array(all_boxes_inst[i][idx + delta])
                keep = nms(all_boxes_inst[i][idx + delta])
                all_boxes[i][idx + delta] = all_boxes_inst[i][idx + delta][keep, :]
        inst_mem.extend(list_new_ginsts)


        out_im_ginst = 0
        out_im_linst = 0
        if vis:
            out_im_ginst = draw_all_ginst(center_image, inst_mem, classes, scales[delta], cfg, idx+delta)
            out_im_linst = draw_all_linst(center_image, inst_mem_now, inst_mem, classes, scales[delta], cfg)
            #out_im2 = draw_all_rois(center_image, rois, scales[delta], cfg) # print rois from RPN

        out_im = 0

    return ginst_ID, out_im, out_im_ginst, out_im_linst
