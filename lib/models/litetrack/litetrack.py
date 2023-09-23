"""
Basic LiteTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.litetrack.vit_cae_async import CAE_Base_patch16_224_Async
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LiteTrack(nn.Module):
    """ This is the base class for LiteTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", aux_template_loss=False,
                 search_feat_size=16, template_feat_size=8, add_target_token=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type

        self.feat_size_s = int(search_feat_size)
        self.feat_len_s = int(search_feat_size ** 2)
        self.feat_size_t = int(template_feat_size)
        self.feat_len_t = int(template_feat_size ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        self.aux_template_loss = aux_template_loss
        self.add_target_token = add_target_token
        if self.add_target_token:
            self.target_token_embed = Mlp(
                4, out_features=self.backbone.embed_dim)
        self.onnx = False
        if self.onnx:
            self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True)

    def forward_z(self, template: torch.Tensor, template_bb=None,):
        target_token = None
        if self.add_target_token:
            target_token = self.target_token_embed(template_bb).unsqueeze(-2)
        return self.backbone(z=template, target_token=target_token, mode='z')

    def forward(self,template_feats, search ):
        x = self.backbone(z=template_feats,x=search, mode='x')
        out = self.forward_head(x)
        if self.onnx:
            response = self.output_window * out['score_map']
            return response, out['size_map'], out['offset_map']
        return out

    def forward_train(self, template: torch.Tensor,
                search: torch.Tensor,
                template_bb=None,
                gt_cls_map=None,
                ):
        if self.add_target_token:
            target_token = self.target_token_embed(template_bb).unsqueeze(-2)
            x= self.backbone(z=template,
                                        x=search,
                                        target_token=target_token
                                        )
        else:
            x = self.backbone(z=template, x=search)
        out = self.forward_head(x, gt_cls_map)

        return out

    def forward_head(self, cat_feature,
                     gt_score_map=None,
                     ):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if self.head_type == "CENTER":
            # encoder output for the search region (B, HW, C)
            enc_opt = cat_feature[:, -self.feat_len_s:]
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)

            score_map_ctr, bbox, size_map, offset_map = self.box_head(
                opt_feat, self.feat_size_s)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}

            return out

        else:
            raise NotImplementedError


def build_LiteTrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(
        __file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('LiteTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_CAE':
        backbone = CAE_Base_patch16_224_Async(pretrained,
                                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              add_cls_token=cfg.MODEL.USE_CLS_TOKEN,
                                              num_async_interaction_stage=cfg.MODEL.BACKBONE.AI_LAYERS,
                                              depth=cfg.MODEL.BACKBONE.DEPTH
                                              )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_size_s = int(cfg.DATA.SEARCH.SIZE / stride)
    feat_size_t = int(cfg.DATA.TEMPLATE.SIZE / stride)
    if not training:
        cfg.MODEL.AUX_TEMPLATE = False
    model = LiteTrack(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD.TYPE,
        search_feat_size=feat_size_s,
        template_feat_size=feat_size_t,
        add_target_token=cfg.MODEL.USE_CLS_TOKEN,
    )

    if 'LiteTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
