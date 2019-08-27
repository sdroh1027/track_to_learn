# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import time

import logging
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
from core.tester import im_detect_all, Predictor, get_resnet_output, prepare_data, draw_all_detection, draw_all_rois, draw_all_instances
from symbols import *
from nms.seq_nms import seq_nms
from utils.load_model import load_param
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

# sidong
from instance import Instance, init_inst_params
import logging
ginst_ID = 0

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


def process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, center_image, scales, embeds, inst_mem, sim_array_global):
    global logger
    global ginst_ID

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

            boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, num_classes)]

            out_im = draw_all_detection(center_image, boxes_this_image, classes, scales[delta], cfg)

            return out_im, out_im2

    return 0,out_im2


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
    all_embeds = [[] for _ in range(300)] #all_embeds = [[] for _ in range(len(data))]

    inst_mem = [] # list for instance class
    sim_array_global = [] # similarity array list

    data_list = deque(maxlen=all_frame_interval)
    feat_list = deque(maxlen=all_frame_interval)
    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
    # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
    while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
        data_list.append(image)
        feat_list.append(feat)

    vis = False
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
                out_im, out_im2 = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes,
                                                    file_idx, max_per_image, vis,
                                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales, all_embeds, inst_mem, sim_array_global)
                #out_im2 = process_pred_result_rois(pred_result, cfg.TEST.RPN_NMS_THRESH, cfg, nms, all_rois, file_idx, max_per_image,
                #                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                ginst_ID_now = ginst_ID
                init_inst_params(inst_mem, ginst_ID_prev, ginst_ID_now, max_inst, aggr_predictors, arg_params)

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
                out_im, out_im2 = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes,
                                                      file_idx, max_per_image, vis,
                                                      data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales,
                                                      all_embeds, inst_mem, sim_array_global)
                # out_im2 = process_pred_result_rois(pred_result, cfg.TEST.RPN_NMS_THRESH, cfg, nms, all_rois, file_idx, max_per_image,
                #                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                ginst_ID_now = ginst_ID
                init_inst_params(inst_mem, ginst_ID_prev, ginst_ID_now, max_inst, aggr_predictors ,arg_params)

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
