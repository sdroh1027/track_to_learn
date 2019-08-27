# --------------------------------------------------------
# Track to learn
# Copyright (c) 2019 ESoC
# Licensed under The MIT License [see LICENSE for details]
# Written by Si-Dong Roh
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import time

import pprint
import cv2
from config.config import config as cfg
from config.config import update_config
from utils.image import resize, transform
import numpy as np
from collections import deque


# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fgfa_rfcn/cfgs/fgfa_rfcn_vid_demo_online_train.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet/', cfg.MXNET_VERSION))
import mxnet as mx
import time
from core.tester import Predictor, get_resnet_output, prepare_data, draw_all_detection
from symbols import *
from nms.seq_nms import seq_nms
from utils.load_model import load_param
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

# sidong
from core.instance import init_inst_params
from core.tester_online import im_detect_all, process_pred_result_ot
import logging
logger = logging.getLogger("lgr")
logger.setLevel(logging.DEBUG) #INFO
stream_hander = logging.StreamHandler()
logger.addHandler(stream_hander)
logger.info("logger start!")

def parse_args():
    parser = argparse.ArgumentParser(description='Show Flow-Guided Feature Aggregation demo')
    args = parser.parse_args()
    return args

args = parse_args()

def save_image(output_dir, count, out_im):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    filename = str(count) + '.JPEG'
    cv2.imwrite(output_dir + filename, out_im)

def main():
    # get symbol
    pprint.pprint(cfg)
    cfg.symbol = 'resnet_v1_101_flownet_rfcn_online_train'
    model = '/../model/rfcn_fgfa_flownet_vid_original'

    all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
    max_per_image = cfg.TEST.max_per_image
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)

    # set up class names
    num_classes = 31
    classes = ['__background__','airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    # load demo data

    snippet_name = 'ILSVRC2015_val_00016002/'# 'ILSVRC2015_val_00044006/' #'ILSVRC2015_val_00007010/' #'ILSVRC2015_val_00016002/'
    image_names = glob.glob(cur_path + '/../demo/' + snippet_name + '/*.JPEG')
    image_names.sort()
    output_dir = cur_path + '/../demo/test_'# rfcn_fgfa_online_train_'
    output_dir_roi = cur_path + '/../demo/test_rois_'  # rfcn_fgfa_online_train_'
    if (cfg.TEST.SEQ_NMS):
        output_dir += 'SEQ_NMS_'
        output_dir_roi += 'SEQ_NMS_'
    output_dir += snippet_name
    output_dir_roi += snippet_name

    data = []
    for im_name in image_names:
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = cfg.SCALES[0][0]
        max_size = cfg.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
        data.append({'data': im_tensor, 'im_info': im_info,  'data_cache': im_tensor,    'feat_cache': im_tensor})



    # get predictor

    print 'get-predictor'
    data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
    label_names = []

    t1 = time.time()
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_cache', (19, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('feat_cache', ((19, cfg.network.FGFA_FEAT_DIM,
                                                np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int))))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for _ in xrange(len(data))]

    arg_params, aux_params = load_param(cur_path + model, 0, process=True)

    #add parameters for instance cls & regression
    arg_params['rfcn_ibbox_bias'] = arg_params['rfcn_bbox_bias'].copy()  #deep copy
    arg_params['rfcn_ibbox_weight'] = arg_params['rfcn_bbox_weight'].copy()  #deep copy
    max_inst = cfg.TEST.NUM_INSTANCES


    feat_predictors = Predictor(feat_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    aggr_predictors = Predictor(aggr_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = py_nms_wrapper(cfg.TEST.NMS)


    # First frame of the video
    idx = 0
    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    all_boxes = [[[] for _ in range(len(data))]
                 for _ in range(num_classes)]

    ginst_mem = [] # list for instance class
    sim_array_global = [] # similarity array list
    ginst_ID = 0

    data_list = deque(maxlen=all_frame_interval)
    feat_list = deque(maxlen=all_frame_interval)
    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
    # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
    while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
        data_list.append(image)
        feat_list.append(feat)

    vis = True
    file_idx = 0
    thresh = 1e-3
    for idx, element in enumerate(data):

        data_batch = mx.io.DataBatch(data=[element], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, element)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        if(idx != len(data)-1):
            if len(data_list) < all_frame_interval - 1:
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

            else:
                #if file_idx ==15:
                #    print '%d frame' % (file_idx)
                #################################################
                # main part of the loop
                #################################################
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

                prepare_data(data_list, feat_list, data_batch) #put 19 data & feat list into data_batch
                #aggr_predictors._mod.forward(data_batch) #sidong
                #zxcv= aggr_predictors._mod.get_outputs(merge_multi_context=False)#sidong
                pred_result = im_detect_all(aggr_predictors, data_batch, data_names, scales, cfg) # get box result [[scores, pred_boxes, rois, data_dict, iscores, ipred_boxes, cropped_embed]]

                data_batch.data[0][-2] = None # 19 frames of data possesses much memory, so clear it
                data_batch.provide_data[0][-2] = ('data_cache', None) # also clear shape info of data
                data_batch.data[0][-1] = None
                data_batch.provide_data[0][-1] = ('feat_cache', None)

                ginst_ID_prev = ginst_ID
                ginst_ID, out_im, out_im2 = process_pred_result_ot(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes,
                                                    file_idx, max_per_image, vis,
                                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales, ginst_mem, sim_array_global, ginst_ID)
                #out_im2 = process_pred_result_rois(pred_result, cfg.TEST.RPN_NMS_THRESH, cfg, nms, all_rois, file_idx, max_per_image,
                #                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                ginst_ID_now = ginst_ID
                init_inst_params(ginst_mem, ginst_ID_prev, ginst_ID_now, max_inst, aggr_predictors, arg_params)

                total_time = time.time()-t1
                if (cfg.TEST.SEQ_NMS==False):
                    save_image(output_dir, file_idx, out_im)
                    save_image(output_dir_roi, file_idx, out_im2)

                #testing by metric


                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time /(file_idx+1))
                file_idx += 1
        else:
            #################################################
            # end part of a video                           #
            #################################################

            end_counter = 0
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                data_list.append(image)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                pred_result = im_detect_all(aggr_predictors, data_batch, data_names, scales, cfg)

                ginst_ID_prev = ginst_ID
                ginst_ID, out_im, out_im2 = process_pred_result_ot(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes,
                                                      file_idx, max_per_image, vis,
                                                      data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales,
                                                      ginst_mem, sim_array_global, ginst_ID)
                # out_im2 = process_pred_result_rois(pred_result, cfg.TEST.RPN_NMS_THRESH, cfg, nms, all_rois, file_idx, max_per_image,
                #                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                ginst_ID_now = ginst_ID
                init_inst_params(ginst_mem, ginst_ID_prev, ginst_ID_now, max_inst, aggr_predictors ,arg_params)

                total_time = time.time() - t1
                if (cfg.TEST.SEQ_NMS == False):
                    save_image(output_dir, file_idx, out_im)
                    save_image(output_dir_roi, file_idx, out_im2)
                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time / (file_idx+1))
                file_idx += 1
                end_counter += 1

    if(cfg.TEST.SEQ_NMS):
        video = [all_boxes[j][:] for j in range(1, num_classes)]
        dets_all = seq_nms(video)
        for cls_ind, dets_cls in enumerate(dets_all):
            for frame_ind, dets in enumerate(dets_cls):
                keep = nms(dets)
                all_boxes[cls_ind + 1][frame_ind] = dets[keep, :]
        for idx in range(len(data)):
            boxes_this_image = [[]] + [all_boxes[j][idx] for j in range(1, num_classes)]
            out_im = draw_all_detection(data[idx][0].asnumpy(), boxes_this_image, classes, scales[0], cfg)
            save_image(output_dir, idx, out_im)

    print 'done'

if __name__ == '__main__':
    main()
