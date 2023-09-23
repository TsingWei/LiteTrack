"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.vit_cae import CAE_Base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh



class ContrastiveEmbed(nn.Module):
    '''
    A classification head using dynamic objects as refference.
    '''
    def __init__(self, max_obj_len=256, num_classes=1, in_dim=768, out_dim=256):
        """
        Args:
            max_obj_len: max length of obj.
        """
        super().__init__()
        self.max_obj_len = max_obj_len
        self.num_classes = num_classes
        self.transform = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, obj):
        """_summary_

        Args:
            x (tensor): unknown obj
            obj (tensor): known obj
        Returns:
            _type_: _description_
        """
        y = obj
        # text_token_mask = text_dict["text_token_mask"]

        res = self.transform(x) @ self.transform(y).transpose(-1, -2)
        # res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        # new_res = torch.full((*res.shape[:-1], self.max_obj_len), float("-inf"), device=res.device, dtype=torch.float16)
        # new_res[..., : res.shape[-1]] = res

        return res


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

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

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
        # feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # if head_type == "CORNER" or head_type == "CENTER":
        self.feat_size_s = int(search_feat_size)
        self.feat_len_s = int(search_feat_size ** 2)
        self.feat_size_t = int(template_feat_size)
        self.feat_len_t = int(template_feat_size ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        self.aux_template_loss = aux_template_loss
        self.add_target_token = add_target_token
        if self.add_target_token:
            self.target_token_embed = Mlp(4, out_features=self.backbone.embed_dim)
        self.onnx = True
        # self.neck = MLP(768,768,256,3)
        # self.prpool = PrRoIPool(1)
        # self.label_enc = nn.Embedding(4 + 1, 768)
        # self.constrative_head = ContrastiveEmbed()

    def forward_z():
        pass

    def forward_x():
        pass
            
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_bb=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                search_bb=None,
                gt_cls_map = None,
                threshold=0,
                gt_cls_map_T=None,
                mask_z=None,
                mask_s=None,
                mode=None,
                temporal_feat=None,
                head_only=False,
                ):
        if False:
            feat, _ = self.backbone(z=template, x=search,
                                ce_template_mask=ce_template_mask,
                                ce_keep_rate=ce_keep_rate,
                                template_bb=template_bb,
                                mode=mode)
            if mode=='z':
                return feat
            return self.forward_head(feat, gt_cls_map,
                                 gt_score_map_T=gt_cls_map_T, 
                                 th=threshold, 
                                 mask_z=mask_z,
                                 mask_s=mask_s,
                                 template_bb=template_bb,
                                 search_bb=search_bb)
            

                
        # if self.add_target_token:
        #     target_token = self.target_token_embed(template_bb).unsqueeze(-2)
        #     # x, aux_dict = self.backbone(z=template, x=search,
        #     #                         ce_template_mask=ce_template_mask,
        #     #                         ce_keep_rate=ce_keep_rate,
        #     #                         return_last_attn=return_last_attn,
        #     #                         template_bb=template_bb,
        #     #                         target_token=target_token)
        # else:

        if self.add_target_token:
            target_token = self.target_token_embed(template_bb).unsqueeze(-2)
            x, aux_dict = self.backbone(z=template, x=search,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            target_token=target_token
                            # return_last_attn=return_last_attn,
                            )
        else:
            x, aux_dict = self.backbone(z=template, x=search,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            # target_token=target_token
                            # return_last_attn=return_last_attn,
                            )
            

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        attn = aux_dict['attn']
        out = {}
        # return
        # size_token = x[:, -1:]
        # x = x[:, :-1]
        # guess_size = self.size_guessor(size_token).squeeze_(1).sigmoid()
        # feat_search = x[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # pooled_feat = self.prpool(feat_search.permute((0, 2, 1)).contiguous().reshape(32,-1,16,16), template_bb)
        # out = {}

        


        out = self.forward_head(x, gt_cls_map,
                                gt_score_map_T=gt_cls_map_T, 
                                th=threshold, wh=None, attn=attn, 
                                mask_z=mask_z,
                                mask_s=mask_s,
                                template_bb=template_bb,
                                search_bb=search_bb,
                                )
        if self.onnx:
            return out['score_map'], out['size_map'], out['offset_map']

        out.update(aux_dict)
        # out['feat_search'] = feat_search
        # out['guess_size'] = guess_size
        return out

    def forward_head(self, cat_feature,
                    gt_score_map=None,
                    th=0, 
                    gt_score_map_T=None,
                    wh=None, attn=None, 
                    mask_z=None, 
                    mask_s=None, 
                    template_bb=None, 
                    search_bb=None,
                    **kwargs,
                    ):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if self.aux_template_loss:
            # TODO if cls token?
            enc_opt_T = cat_feature[:, :self.feat_len_t]  # encoder output for the search region (B, HW, C)
            opt_T = (enc_opt_T.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt_T.size()
            opt_feat_T = opt_T.view(-1, C, self.feat_size_t, self.feat_size_t)
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, self.feat_size_s, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            if self.aux_template_loss:
                # shared center head
                score_map_ctr_T, bbox_T, size_map_T, offset_map_T = self.box_head(opt_feat_T, self.feat_size_t,)
                outputs_coord_T = bbox_T
                outputs_coord_new_T = outputs_coord_T.view(bs, Nq, 4)
                out['pred_boxes_T'] = outputs_coord_new_T
                out['score_map_T'] = score_map_ctr_T
                out['size_map_T'] = size_map_T
                out['offset_map_T'] = offset_map_T
        
            return out
        elif self.head_type == "GFL":
            # run the head

            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)
            box_best = None
            score_map_ctr, bboxes, ltrb_distribution, box_best, qs = self.box_head(opt_feat, self.feat_size_s, gt_score_map, th=th)
            outputs_coord = box_xyxy_to_cxcywh(bboxes)
            # bboxes = box_xyxy_to_cxcywh(bboxes)

            # Pooling
            # batch_index = torch.arange(bs, dtype=torch.float32).view(-1, 1).to(template_bb.device)
            # box_z = template_bb * 16
            # pooled_z = self.prpool(feat_t, torch.cat((batch_index, box_z), dim=1)).flatten(2).permute((0, 2, 1))
            # box_x = box_best * 16
            # pooled_x = self.prpool(feat_s, torch.cat((batch_index, box_x), dim=1)).flatten(2).permute((0, 2, 1))
            # box_x_GT = search_bb *16
            # pooled_x_GT = self.prpool(feat_s, torch.cat((batch_index, box_x_GT), dim=1)).flatten(2).permute((0, 2, 1))
            # corr = self.constrative_head(pooled_z, pooled_x)
            # corr_GT = self.constrative_head(pooled_z, pooled_x_GT)

            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord,
                #    'score_map_c': score_map_c,
                   'score_map': score_map_ctr,
                   'ltrb_dis': ltrb_distribution,
                   'box_best': box_best,
                   'boxes': bboxes,
                   'qs': qs,
                #    'corr': corr,
                #    'corr2GT': corr_GT,
                   }
            if self.aux_template_loss:
                # shared center head
                box_best_T = None
                score_map_ctr_T, bboxes_T, ltrb_distribution_T, box_best_T = self.box_head(opt_feat_T, self.feat_size_t, gt_score_map_T, th=th)
                outputs_coord_T = box_xyxy_to_cxcywh(bboxes_T)
                # outputs_coord_new_T = outputs_coord_T.view(bs, Nq, 4)
                out['pred_boxes_T'] = outputs_coord_T
                out['score_map_T'] = score_map_ctr_T
                out['ltrb_dis_T'] = ltrb_distribution_T
                out['box_best_T'] = box_best_T
                out['boxes_T'] = bboxes_T
            return out
        elif self.head_type == "PyGFL":
            # run the head
            box_best = None
            score_map_ctr, bboxes, ltrb_distribution, box_best = self.box_head(cat_feature, self.feat_size_s, gt_score_map, th=th)
            outputs_coord = box_xyxy_to_cxcywh(bboxes)
            # bboxes = box_xyxy_to_cxcywh(bboxes)

            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord,
                   'score_map': score_map_ctr,
                   'ltrb_dis': ltrb_distribution,
                   'box_best': box_best,
                   'boxes': bboxes
                   }
            return out
        elif self.head_type == "REPCENTER":
            # run the head
            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)
            box_best = None
            cls_map, bboxes, _, box_best = self.box_head(opt_feat, self.feat_size_s, gt_score_map, th=th)
            outputs_coord = bboxes
            # bboxes = box_xyxy_to_cxcywh(bboxes)

            # outputs_coord = bbox
            # outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord,
                   'score_map': cls_map,
                #    'ltrb_dis': ltrb_distribution,
                   'box_best': box_best,
                   'boxes': bboxes
                   }
            return out
        elif self.head_type == "REPPOINTS":
            # run the head
            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_size_s, self.feat_size_s)
            box_best = None
            # box_in = torch.cat((-wh*0.5, wh*0.5), -1) * self.feat_size_s 
            # box_in = box_in[...,None,None].repeat(1,1,16,16).cuda()
            cls_score, init_box, refine_box, init_pts, refine_pts, anchor_box = self.box_head(opt_feat, self.feat_size_s, gt_score_map, th=th, box_in=None)
            # xyxy
            out = { 'score_map': cls_score,
                    'init_box': init_box,
                    'refine_box': refine_box,
                    'init_pts': init_pts,
                    'refine_pts': refine_pts,
                    'anchor_box': anchor_box,
                   }
            return out
        elif self.head_type == "DECODER":

            # cdn
            

            # run the head
            # cat_feature = self.neck(cat_feature)

            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            enc_opt_T = cat_feature[:, :self.feat_len_t].to(torch.float32)
            
            # pooled_feat = self.prpool(enc_opt.reshape(32,16,16,-1), )
            # 
            
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            feat_s = opt.view(-1, C, self.feat_size_s, self.feat_size_s)
            feat_t = enc_opt_T.permute((0, 2, 1)).contiguous().reshape(bs,-1,self.feat_size_t,self.feat_size_t)
            

            box_best = None
            out_dict={}
            out_dict = self.box_head(feat_s, feat_t, attn, mask_z, mask_s, template_bb, search_bb, **kwargs,)
            # out_dict['template_tokens'] = template_feat
            # # xyxy
            # out = { 'score_map': cls_score,
            #         'init_box': init_box,
            #         'refine_box': refine_box,
            #         'init_pts': init_pts,
            #         'refine_pts': refine_pts,
            #         'anchor_box': anchor_box,
            #        }
            return out_dict
        
        else:
            raise NotImplementedError

class SimpleFPN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # # Up sampling 4X
        # self.fpn1 = nn.Sequential(
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #     Norm2d(embed_dim),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        # )

        # Up sampling 2X
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        # Keep 1X
        self.fpn3 = nn.Identity()

        # Down sampling 0.5X
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

    
    def forward(self, x):
        B, C, H, W = x.shape
        xp = x.permute(0, 2, 1).reshape(B, -1, H, W)
        features = []
        ops = [ self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            features.append(ops[i](xp))
        return features

def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_CAE':
        backbone = CAE_Base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                        CE=cfg.MODEL.BACKBONE.CE,
                                        add_cls_token = cfg.MODEL.USE_CLS_TOKEN,
                                        seperate_loc = cfg.MODEL.BACKBONE.SEP_LOC,
                                        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1


    elif cfg.MODEL.BACKBONE.TYPE == 'cae_fpn':
        backbone = cae_fpn(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'swin':
        backbone = SwinBackbone4Track(pretrained)
        hidden_dim = backbone.embed_dim
        patch_start_index = 0

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
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
    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        search_feat_size=feat_size_s,
        template_feat_size=feat_size_t,
        aux_template_loss=cfg.MODEL.AUX_TEMPLATE,
        add_target_token=cfg.MODEL.USE_CLS_TOKEN,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
