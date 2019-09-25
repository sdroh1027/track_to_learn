# --------------------------------------------------------
# Instance linking
# Copyright ESoC
# Licensed under The MIT License [see LICENSE for details]
# Written by Si-Dong Roh
# --------------------------------------------------------
from bbox.bbox_transform import clip_boxes
import logging
logger = logging.getLogger("lgr")
class Instance:
    global logger
    def __init__(self, GID = None, LID=None, frame_idx = None, cls = None, cls_score = None , bbox = None,  embed_feat = None, atten = None):
        assert frame_idx >= 0
        self.GID = GID # it means its global ID (only 1L tuple)
        self.LID = LID
        self.cls = cls
        self.cls_score = cls_score
        self.cls_score_reliable = cls_score
        self.cls_high = cls
        self.cls_score_high = cls_score
        self.bbox = bbox
        self.embed_feat = embed_feat
        self.atten = atten
        self.detected_idx = [frame_idx]
        self.color = None
        self.hard_case = False
        self.linked_to = [] # for local inst, it means linked global IDs (can be multiple)
        self.linked_to_gidx = None
        self.make_second_bbox()
        self.l_suppressed = []

        self.cls_score_list = []
        self.cls_list = []

    def update_manual(self, frame_idx, cls_score, bbox, embed_feat, atten):
        self.cls_score = cls_score
        self.bbox = bbox
        self.make_second_bbox() # center, bbox2 created
        self.embed_feat = embed_feat
        self.atten = atten
        self.detected_idx.append(frame_idx)

    def update_from_inst(self, new_inst = None):
        self.cls = new_inst.cls
        self.cls_score = new_inst.cls_score
        self.bbox = new_inst.bbox
        self.center = new_inst.center
        self.bbox2 = new_inst.bbox2
        self.embed_feat = new_inst.embed_feat
        self.atten = new_inst.atten
        self.detected_idx.append(new_inst.detected_idx[-1])
        print 'new_inst added'

    def update_intra_frame(self, new_inst=None):
        if new_inst.cls_score > self.cls_score:
            self.cls = new_inst.cls
            self.cls_score = new_inst.cls_score
            self.bbox = new_inst.bbox
            self.center = new_inst.center
            self.bbox2 = new_inst.bbox2
            self.embed_feat = new_inst.embed_feat
            self.atten = new_inst.atten
            self.detected_idx.append(new_inst.detected_idx[-1])
            logger.debug('changed to new_inst because of higher cls_score')
        else:
             logger.debug('inst not changed because of lower cls_score')

    def update_inter_frame(self, new_inst=None, sim=None):
        self.sim = sim
        self.cls_score_reliable = max(sim * self.cls_score_reliable, new_inst.cls_score)
        self.LID = new_inst.LID
        self.cls = new_inst.cls
        self.cls_score = new_inst.cls_score
        self.bbox = new_inst.bbox
        self.center = new_inst.center
        self.bbox2 = new_inst.bbox2
        self.embed_feat = new_inst.embed_feat
        self.atten = new_inst.atten
        self.sim = sim
        self.detected_idx.append(new_inst.detected_idx[-1])
        self.cls_list.append(new_inst.cls)
        self.cls_score_list.append(new_inst.cls_score)
        self.cls_score_acc += self.cls_score
        # save linked global IDs to local inst
        new_inst.linked_to.append(self.GID)

        if new_inst.cls_score > self.cls_score_high:
            if self.cls_high != new_inst.cls:
                logger.debug('class_high is changed to latest one because of higher cls_score')
            self.cls_high = new_inst.cls
            self.cls_score_high = new_inst.cls_score

    def make_second_bbox(self):
        # make virtual bbox for inter frame iou calculation
        self.center = [(self.bbox[0] + self.bbox[2])/2, (self.bbox[1] + self.bbox[3])/2]
        self.bbox2 = self.bbox.copy()
        w = max(0.0, self.bbox[2] - self.bbox[0] + 1)
        h = max(0.0, self.bbox[3] - self.bbox[1] + 1)
        d = abs((w - h) / 2)

        if w>h:
            self.bbox2[1] = self.bbox[1] - d
            self.bbox2[3] = self.bbox[3] + d
        else:
            self.bbox2[0] = self.bbox[0] - d
            self.bbox2[2] = self.bbox[2] + d

    def make_global_inst(self, GID):
        assert GID is not None
        ginst = Instance(GID=GID, LID=self.LID, frame_idx=self.detected_idx[-1], cls=self.cls, cls_score=self.cls_score,
                         bbox=self.bbox, embed_feat=self.embed_feat, atten=self.atten)
        ginst.make_second_bbox()
        ginst.g_suppressed = []
        ginst.linked_to_LID = None
        ginst.sim = 0
        ginst.cls_high = self.cls
        ginst.cls_score_high = self.cls_score
        ginst.cls_score_reliable = self.cls_score
        ginst.cls_score_list = [self.cls_score]
        ginst.cls_list = [self.cls]
        ginst.cls_score_acc = self.cls_score
        # save linked global IDs to local inst
        self.linked_to.append(self.GID)
        return ginst

def init_inst_params(inst_mem, GID_prev, ginst_ID_now, max_inst, aggr_predictors, arg_params):
    if (GID_prev < max_inst):
        to_ = min(ginst_ID_now,8)
        for ginst in inst_mem[GID_prev:to_]:
            # get weight of ginst.cls_high
            print 'initializing parameter for GID:%d' % ginst.GID
            index_from = 49 * ginst.cls_high
            index_to =  index_from + 49
            arg_params["rfcn_inst_%d_weight" % ginst.GID] = arg_params['rfcn_cls_weight'][index_from:index_to].copy()  # deep copy
            arg_params['rfcn_inst_%d_bias' % ginst.GID] = arg_params['rfcn_cls_bias'][index_from:index_to].copy()  # deep copy
            aggr_predictors._mod.init_params(arg_params=arg_params, aux_params=[], allow_missing=True, force_init= True)
    else:
        print 'inst parameter not updated'