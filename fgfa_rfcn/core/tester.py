# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

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

#from utils.plot_tensor import plot_tensor


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        # [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict, scale) in enumerate(zip(scores_all, boxes_all, data_dict_all, scales)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def get_resnet_output(predictor, data_batch, data_names):
    output_all = predictor.predict(data_batch)
    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]

    if output_all[0].has_key('conv_embed_output'):
        feat = output_all[0]['conv_embed_output']
    else:
        feat = None
    return data_dict_all[0]['data'], feat.copy()


def im_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)
    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return zip(scores_all, pred_boxes_all, data_dict_all)

def im_batch_detect(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        im_infos = data_dict['im_info'].asnumpy()
        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
        rois = output['rois_output'].asnumpy()
        for im_idx in xrange(im_infos.shape[0]):
            bb_idxs = np.where(rois[:,0] == im_idx)[0]
            im_shape = im_infos[im_idx, :2].astype(np.int)

            # post processing
            pred_boxes = bbox_pred(rois[bb_idxs, 1:], bbox_deltas[bb_idxs, :])
            pred_boxes = clip_boxes(pred_boxes, im_shape)

            # we used scaled image & roi to train, so it is necessary to transform them back
            pred_boxes = pred_boxes / scale[im_idx]

            scores_all.append(scores[bb_idxs, :])
            pred_boxes_all.append(pred_boxes)

    return scores_all, pred_boxes_all, data_dict_all

def pred_eval_seqnms(gpu_id,imdb):

    det_file = os.path.join(imdb.result_path, imdb.name + '_' + str(gpu_id) + '_raw')
    print 'det_file=', det_file
    if os.path.exists(det_file):
        with open(det_file, 'rb') as fid:
            all_boxes, frame_ids = dill.load(fid)

        res=[all_boxes, frame_ids]
        imdb.evaluate_detections_multiprocess_seqnms(res, gpu_id)

def pred_eval(gpu_id, feat_predictors, aggr_predictors, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
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
                pred_result = im_detect(aggr_predictors, data_batch, data_names, scales, cfg) #it returns zip(scores_all, pred_boxes_all, rois_all, data_dict_all, iscores_all, ipred_boxes_all, cropped_embeds_all)

                roidb_offset += 1
                frame_ids[idx] = roidb_frame_ids[roidb_idx] + roidb_offset

                t2 = time.time() - t
                t = time.time()
                process_pred_result(pred_result, imdb, thresh, cfg, nms, all_boxes, idx, max_per_image, vis,
                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
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
                pred_result = im_detect(aggr_predictors, data_batch, data_names, scales, cfg)

                roidb_offset += 1
                frame_ids[idx] = roidb_frame_ids[roidb_idx] + roidb_offset

                t2 = time.time() - t
                t = time.time()
                process_pred_result(pred_result, imdb, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
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

def run_dill_encode(payload):
    fun,args=dill.loads(payload)
    return fun(*args)
    
def apply_async(pool,fun,args):
    payload=dill.dumps((fun,args))
    return pool.apply_async(run_dill_encode,(payload,))

def pred_eval_multiprocess(gpu_num, key_predictors, cur_predictors, test_datas, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):

    if cfg.TEST.SEQ_NMS==False:
        if gpu_num == 1:
            res = [pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis, thresh, logger,
                             ignore_cache), ]
        else:
            from multiprocessing.pool import ThreadPool as Pool
            pool = Pool(processes=gpu_num)
            multiple_results = [pool.apply_async(pred_eval, args=(
            i, key_predictors[i], cur_predictors[i], test_datas[i], imdb, cfg, vis, thresh, logger, ignore_cache)) for i
                                in range(gpu_num)]
            pool.close()
            pool.join()
            res = [res.get() for res in multiple_results]
        info_str = imdb.evaluate_detections_multiprocess(res)


    else :
        if gpu_num == 1:
            res = [pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis, thresh, logger, ignore_cache),]

        else:
            from multiprocessing.pool import ThreadPool as Pool

            pool = Pool(processes=gpu_num)
            multiple_results = [pool.apply_async(pred_eval, args=(
            i, key_predictors[i], cur_predictors[i], test_datas[i], imdb, cfg, vis, thresh, logger, ignore_cache)) for i in
                                range(gpu_num)]
            pool.close()
            pool.join()
            res = [res.get() for res in multiple_results]


        from multiprocessing import Pool as Pool
        pool = Pool(processes=gpu_num)
        jobs = []
        res=[]
        for i in range(gpu_num):
            job = apply_async(pool, pred_eval_seqnms, (i, imdb))
            jobs.append(job)
        for job in jobs:
            res.append(job.get())
        info_str = imdb.do_python_eval_gen(gpu_num)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=0.1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=0.1):
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
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im


def prepare_data(data_list, feat_list, data_batch):
    concat_feat = mx.ndarray.concatenate(list(feat_list), axis=0)
    concat_data = mx.ndarray.concatenate(list(data_list), axis=0)
    # data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
    data_batch.data[0][-2] = concat_data
    data_batch.provide_data[0][-2] = ('data_cache', concat_data.shape)
    data_batch.data[0][-1] = concat_feat
    data_batch.provide_data[0][-1] = ('feat_cache', concat_feat.shape)


def process_pred_result(pred_result, imdb, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, center_image, scales):
    for delta, (scores, boxes, data_dict) in enumerate(pred_result):
        for j in range(1, imdb.num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][idx+delta]=cls_dets
            else:
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]

        if cfg.TEST.SEQ_NMS==False and  max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][idx + delta][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][idx + delta][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx + delta] = all_boxes[j][idx + delta][keep, :]

        if vis:
            boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, imdb.num_classes)]
            vis_all_detection(center_image, boxes_this_image, imdb.classes, scales[delta], cfg)
