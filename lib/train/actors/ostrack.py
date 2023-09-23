from . import BaseActor
from lib.utils.misc import NestedTensor, inverse_sigmoid
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou, box_iou_pairwise, xywh_to_cxcywh, box_xyxy_to_xywh
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap, generate_cls_map, generate_distribution_heatmap, grid_center_2d, grid_center_flattened, bbox2distance, get_2d_gaussian_map
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate, generate_mask_cond_search
from ...utils.cdn import prepare_for_cdn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist

def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.reduce_op.SUM)
    return tensor

class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.search_feat_size = self.cfg.DATA.SEARCH.SIZE//self.cfg.MODEL.BACKBONE.STRIDE
        self.template_feat_size = self.cfg.DATA.TEMPLATE.SIZE//self.cfg.MODEL.BACKBONE.STRIDE
        self.count = 0

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        self.count+=1
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1

        template_list = []
        search_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:]).clamp(min=-5.0,max=5.0)  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:]).clamp(min=-5.0,max=5.0)  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            search_list.append(search_img_i)
        # search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:]).clamp(min=-5.0,max=5.0)  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE:
            ce_box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        template_data = data['template_anno'][0]
        template_bb = box_xywh_to_xyxy(template_data).clamp(min=0.0,max=1.0)
        search_data = data['search_anno'] # [n, bs, 4]
        search_bb = box_xywh_to_xyxy(search_data).clamp(min=0.0,max=1.0)

        # if self.cfg.MODEL.USE_CLS_TOKEN:


        if len(template_list) == 1:
            template_list = template_list[0]
        
        
        gt_cls_maps = generate_cls_map(box_xywh_to_xyxy(data['search_anno'][-1]), self.search_feat_size)
        gt_cls_maps_T = generate_cls_map(box_xywh_to_xyxy(data['template_anno'][-1]), self.template_feat_size).to(torch.int).flatten(1)
        # gt_cls_maps = torch.ones_like(gt_cls_maps)
        # if self.cfg.MODEL.HEAD.TYPE == 'REPCENTER':
        gt_cls_maps[:,:,:] = 1
        gaussian_center_map = generate_heatmap(data['search_anno'], self.search_feat_size)[0]
        if len(search_list) == 1:
            search_list = search_list[0]
            search_bb = search_bb[0]
            out_dict = self.net.forward_train(
                                template=template_list,
                                search=search_list,
                                template_bb=template_bb
                                )
        else:
        # online training
            feat_temperal = None
            out_dict_multiple = []
            for search_img, search_bb_single in zip(search_list, search_bb):
                out_dict = self.net(template=template_list,
                                search=search_img,
                                ce_template_mask=ce_box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False,
                                gt_cls_map=gt_cls_maps,
                                template_bb=template_bb,
                                search_bb=search_bb_single,
                                gt_cls_map_T=None,
                                mask_z=gt_cls_maps_T,
                                mask_s=None,
                                head_only=True,
                                temporal_feat=feat_temperal
                                )
                feat_temperal = out_dict['feat'].detach()
                out_dict_multiple.append(out_dict)
            return out_dict_multiple
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if isinstance(pred_dict, list):
            losses=[]
            for i, pred in enumerate(pred_dict):
                gt_dict_single = {k: v[i][None,...] for k, v in gt_dict.items() if k == "search_anno"}
                loss, status = self.compute_losses_DECODER(pred, gt_dict_single,)
                losses.append(loss)
            loss = sum(losses)

        elif self.cfg.MODEL.HEAD.TYPE == 'GFL' or self.cfg.MODEL.HEAD.TYPE == 'PyGFL':
            loss, status = self.compute_losses_GFL(pred_dict, gt_dict,)
        elif self.cfg.MODEL.HEAD.TYPE == 'REPPOINTS':
            loss, status = self.compute_losses_REPPOINTS(pred_dict, gt_dict,)
        elif self.cfg.MODEL.HEAD.TYPE == 'REPCENTER':
            loss, status = self.compute_losses_REPPOINTS_(pred_dict, gt_dict,)
        elif self.cfg.MODEL.HEAD.TYPE == 'DECODER':
            loss, status = self.compute_losses_DECODER(pred_dict, gt_dict,)
        elif self.cfg.MODEL.HEAD.TYPE == 'CENTER':
            loss, status = self.compute_losses_CENTER(pred_dict, gt_dict,)
        if return_status:
            return loss, status
        else:
            return loss
    def compute_losses_CENTER(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        giou_loss = giou_loss.mean()
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
    def compute_losses_GFL_New(self, pred_dict, gt_dict):
        # GT
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0)
        gt_box_vec_xyxy = gt_bbox_xyxy.unsqueeze(1).repeat(1,self.search_feat_size*self.search_feat_size,1).view(-1,4)

        # Predictions
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes = pred_boxes.view(self.bs, -1, 4)
        pred_logits = pred_dict['score_map'] # bs, H, W, class
        best_boxes = pred_dict['box_best'] 
        
        pred_logits_vec = pred_logits.permute(0, 2, 3,
                                      1).reshape(-1, 1)
                                           
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        # Fetch the box with max pred score. 
        pred_box_final_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        pred_logits_flatten = pred_logits.permute(0, 2, 3,1).flatten(1)
        pred_box_pos = torch.gather(pred_box_final_xyxy,1,pred_logits_flatten.max(1, keepdim=True)[1][...,None].expand(-1,-1,4)).view(-1,4)

        # LOC Loss
        pred_giou, pred_iou = self.objective['giou'](pred_box_pos, gt_bbox_xyxy)  # (BN,4) (BN,4)
        l1 = self.objective['l1'](pred_box_pos, gt_bbox_xyxy, reduction='none')  # (BN,4) (BN,4)
        giou_loss = pred_giou.mean()
        l1_loss = l1.mean()

        # CLASS Loss
        pred_boxes_vec_xyxy = pred_box_final_xyxy.view(-1, 4)
        iou, _ = box_iou(pred_boxes_vec_xyxy, gt_box_vec_xyxy)
        gt_inbox_map =  generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.search_feat_size)
        qfloss = self.objective['qfl'](pred_logits_vec, gt_inbox_map.reshape(-1), iou.detach())
        ce_loss = qfloss.mean()

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss\
            + self.loss_weight['l1'] * l1_loss\
            + ce_loss\
            # + df_loss * 0.25\
            # + pool_loss * 0.25
            # + qs_loss * 0.25
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = pred_iou.mean().detach()
        mean_giou_loss = giou_loss.detach()
        ce_loss_ = ce_loss.detach()
        
        status = {"Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/l1": l1_loss.item(),
                    # "Loss/dfl": df_loss.item(),
                    "Loss/qfl": ce_loss_.item(),
                    "IoU": mean_iou.item(),
                    }
        return loss, status



    def compute_losses_GFL(self, pred_dict, gt_dict):
        # Get GT boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # Get pred
        pred_boxes = pred_dict['pred_boxes']
        # pred_center_score_unsig = pred_dict['score_map_c']
        # qs = pred_dict['qs']
        pred_quality_score = pred_dict['score_map']
        best_boxes = pred_dict['box_best']
        # attn = pred_dict['attn'] # bs, nhead, (81+256), (81+256)
        # corr2GT = pred_dict['corr2GT']
        

        q_score = pred_quality_score.permute(0, 2, 3,
                                      1).reshape(-1, 1)
                                           
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        
        gt_inbox_map = generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.search_feat_size)
        gt_center_map = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)[0]
        zero_map = gt_inbox_map.new_zeros(gt_inbox_map.flatten(1).shape)

        filtered_pred_init, filtered_gt_init, box_pos_index = filter_predictions_by_iou(gt_bbox, pred_boxes.view(self.bs, -1, 4), iou_threshold=0.7, k=1, maxk=100)
        filtered_box_map = zero_map.clone().flatten(0).scatter_(0, box_pos_index.cuda(), 1)
        gt_inbox_map_box = gt_inbox_map #* filtered_box_map.reshape(gt_inbox_map.shape)

        if self.cfg.TRAIN.TRAIN_CLS:
            gt_inbox_map = gt_inbox_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
                                self.search_feat_size,self.search_feat_size)
            gt_center_map = gt_center_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
                                self.search_feat_size,self.search_feat_size)
            # topk_map_for_box = topk_map_for_box * gt_dict['label'].view(-1)[:,None,None].repeat(1,
            #                     self.search_feat_size,self.search_feat_size)
            # topk_map_for_cls = topk_map_for_cls * gt_dict['label'].view(-1)[:,None,None].repeat(1,
            #                     self.search_feat_size,self.search_feat_size)
        
        # Duplicate GT box
        pos_box_ind = torch.nonzero(gt_inbox_map_box.flatten(1))
        gt_bboxes_multiple = gt_bbox[pos_box_ind[:,0],:]
        pred_boxes_slected = pred_boxes.reshape(self.bs, -1, 4)[pos_box_ind[:,0],pos_box_ind[:,1],:]
        pred_ltrb = pred_dict['ltrb_dis'] # already inside the GT box
        pred_ltrb_pos = pred_ltrb.reshape(self.bs, -1, 68)[pos_box_ind[:,0],pos_box_ind[:,1],:].reshape(-1, 17)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bboxes_multiple).clamp(min=0.0,max=1.0)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes_slected).view(-1, 4)

        iou, _ = box_iou(pred_boxes_vec, gt_boxes_vec)
        iou = iou.detach()
        q_score_pos = pred_quality_score.squeeze(1).flatten(1)[pos_box_ind[:,0], pos_box_ind[:,1]] # BUG done
        

        # # Adaptive select the pos sample
        # pos_ind, selected_iou, select_mask = filter_scores(iou, inside_ind, bs=bs) # ([bs_ind, flatten_map_ind])
        # q_score_pos_unsig = q_score_unsig.flatten(1)[pos_ind[:,0], pos_ind[:,0][1]]
        # pred_box_pos = pred_boxes_vec[select_mask]
        # target_box_pos = gt_boxes_vec[select_mask]
        
        if self.cfg.TRAIN.TRAIN_CLS:
            nonzero_indices=torch.nonzero(gt_dict['label'].view(-1))
            gt_bbox = gt_bbox[nonzero_indices[:,0],:]
            best_boxes = best_boxes[nonzero_indices[:,0],:]

        top_gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0)
        top_pred_boxes_vec = best_boxes

        giou_loss, _ = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        # giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        top_giou, top_iou = self.objective['giou'](top_pred_boxes_vec, top_gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec, reduction='none')  # (BN,4) (BN,4)

        # Map loss
        iou_score = zero_map.clone().flatten(1) 
        iou_score[pos_box_ind[:,0], pos_box_ind[:, 1]] = iou

        # Only fetch selected samples.
        
        target_ltrb_pos =  bbox2distance(grid_center_2d(gt_inbox_map_box, self.search_feat_size),
            gt_boxes_vec * self.search_feat_size, max_dis=self.search_feat_size).reshape(-1)# dont norm to 1

        # The weight for localization, from predicted cls socre.
        target_weight = q_score_pos.detach()

        # Map for classification.
        gaussian_center_map = get_2d_gaussian_map(xywh_to_cxcywh(gt_dict['search_anno'][0]), self.search_feat_size)
        iou_gaussian = iou_score * gaussian_center_map.flatten(1)
        # target_gaussian_weight = gaussian_center_map.flatten(1)[inside_ind[:,0],inside_ind[:,1]] * target_weight
        # gt_gaussian_maps[-1].unsqueeze(1)

        # Focal loss
        # qf_loss = self.objective['dfocal'](inverse_sigmoid(q_score), topk_map_for_cls.reshape(-1,1))
        # QFL handle all the points inside the box.
        # qs_loss = self.objective['qfl'](qs.reshape(-1, 1), gt_q_map.reshape(-1), iou_score.reshape(-1))
        qf_loss = self.objective['qfl'](inverse_sigmoid(q_score), gt_inbox_map.reshape(-1), iou_score.reshape(-1))
        qf_loss = qf_loss.reshape(self.bs,-1).mean(1).sum()
        # DFL only handle the selected points inside the box
        df_loss = self.objective['dfl'](pred_ltrb_pos, target_ltrb_pos, weight=target_weight[:, None].expand(-1, 4).reshape(-1))
        giou_loss = (giou_loss * target_weight).sum()/self.bs # weighted GIoU
        l1_loss = (l1_loss.mean(1) * target_weight).sum()/self.bs # weighted L1
                
        # corr loss
        # pool_loss = self.objective['bce'](corr2GT.view(-1), gt_dict['label'].view(-1))

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss\
            + self.loss_weight['l1'] * l1_loss\
            + qf_loss\
            + df_loss * 0.25\
            # + pool_loss * 0.25
            # + qs_loss * 0.25
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = top_iou.detach().mean()
        mean_giou_loss = top_giou.detach().mean()
        
        status = {"Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/l1": l1_loss.item(),
                #   "Loss/ctr": qs_loss.item(),
                    "Loss/dfl": df_loss.item(),
                    "Loss/qfl": qf_loss.item(),
                    # "Loss/pool": pool_loss.item(),
                    "IoU": mean_iou.item(),
                #   "IoU_T": mean_iou.item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status
    def compute_losses_REPPOINTS_(self, pred_dict, gt_dict):
        # Get GT boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gaussian_center_map = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)[0]
        center_ind_batch, center_ind = torch.where(gaussian_center_map.flatten(1)>=1)
        # Get pred
        pred_box_init_vec = pred_dict['init_box']       # xyxy
        pred_box_refine_vec = pred_dict['refine_box']   # xyxy
        pred_cls_score = pred_dict['score_map']
        bs = pred_cls_score.shape[0]

        pred_box_init = pred_box_init_vec.reshape(bs,-1,4)
        pred_box_refine = pred_box_refine_vec.reshape(bs,-1,4)
        pred_cls_score = pred_cls_score.permute(0, 2, 3,
                                      1).reshape(-1, 1) # (n,1)
        num_queries = pred_box_refine.shape[1]
                                           
        if torch.isnan(pred_box_refine).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        # Calc IoU to select the positive samples for loc(refine).
        gt_boxes_vec_all = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4) --> (n,4)
        iou, _ = box_iou(pred_box_init_vec, gt_boxes_vec_all)
        iou = iou.detach()

        pos_inds = torch.where(iou > 0.5)[0]

        pred_box_refine_pos = pred_box_refine_vec[pos_inds]
        gt_box_pos = gt_boxes_vec_all[pos_inds]


        pred_box_init_center = pred_box_init[center_ind_batch, center_ind]
        pred_box_refine_center = pred_box_refine[center_ind_batch, center_ind]
        gt_boxes_vec_center = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)


        giou_loss_init_center, iou_init_center = self.objective['giou'](pred_box_init_center, gt_boxes_vec_center)
        giou_loss_refine_center, iou_refine_center = self.objective['giou'](pred_box_refine_center, gt_boxes_vec_center)
        giou_loss_refine_pos, _ = self.objective['giou'](pred_box_refine_pos, gt_box_pos)
        l1_loss_center = self.objective['l1'](pred_box_refine_center, gt_boxes_vec_center)  # (BN,4) (BN,4)

        # Map loss
        qf_loss = self.objective['qfl'](pred_cls_score, gaussian_center_map.reshape(-1), gaussian_center_map.reshape(-1))
        qf_loss = qf_loss.mean()
        # DFL only handle the selected points inside the box
        # df_loss = self.objective['dfl'](pred_ltrb_pos, target_ltrb_pos, weight=target_weight[:, None].expand(-1, 4).reshape(-1))
        # giou_loss = (giou_loss * target_weight).mean() # weighted GIoU
        giou_loss_init_center = giou_loss_init_center.mean()
        giou_loss_refine_center = giou_loss_refine_center.mean()
        giou_loss_refine_pos = giou_loss_refine_pos.mean()
                
        # weighted sum
        loss = giou_loss_init_center * 2.0\
        + giou_loss_refine_center * 2.0\
        + giou_loss_refine_pos * 2.0\
        + qf_loss\
        + l1_loss_center * 5.0
            # + df_loss * 0.25\
            # + qs_loss * 0.25
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = iou_refine_center.detach().mean()
        mean_iou_init = iou_init_center.detach().mean()
        mean_giou_loss = giou_loss_refine_center.detach()
        
        status = {"Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/l1": l1_loss_center.item(),
                #   "Loss/ctr": qs_loss.item(),
                    # "Loss/dfl": df_loss.item(),
                    "Loss/qfl": qf_loss.item(),
                    "IoU": mean_iou.item(),
                    "IoU_init": mean_iou_init.item(),
                #   "IoU_T": mean_iou.item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status
    def compute_losses_REPPOINTS(self, pred_dict, gt_dict):
        # Get GT boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gaussian_center_map = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)[0]
        center_ind_batch, center_ind = torch.where(gaussian_center_map.flatten(1)>=1)
        # Get pred
        pred_box_init_vec = pred_dict['init_box']       # xyxy
        pred_box_refine_vec = pred_dict['refine_box']   # xyxy
        pred_cls_score = pred_dict['score_map']
        # anchor_box_vec = pred_dict['anchor_box']
        # guess_size = pred_dict['guess_size']
        bs = pred_cls_score.shape[0]

        pred_box_init = pred_box_init_vec.reshape(bs,-1,4)
        pred_box_refine = pred_box_refine_vec.reshape(bs,-1,4)
        # anchor_box = anchor_box_vec.reshape(bs,-1,4)
        pred_cls_score = pred_cls_score.permute(0, 2, 3,
                                      1).squeeze(-1) # (n,1)
        pred_best_ind = pred_cls_score.view(self.bs, -1).max(1, keepdim=True)[1]
        pred_cls_score_vec = pred_cls_score.reshape(-1,1)
        num_queries = pred_box_refine.shape[1]
                                           
        if torch.isnan(pred_box_refine).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        # Calc IoU to select the positive samples for loc(refine).
        gt_boxes_vec_all = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4) --> (n,4)
        iou, _ = box_iou(pred_box_init_vec, gt_boxes_vec_all)
        iou = iou.detach()

        pos_inds = torch.where(iou > 0.5)[0]

        pred_box_refine_pos = pred_box_refine_vec[pos_inds]
        gt_box_pos = gt_boxes_vec_all[pos_inds]


        pred_box_init_center = pred_box_init[center_ind_batch, center_ind]
        pred_box_refine_center = pred_box_refine[center_ind_batch, center_ind]
        # pred_box_init_center = torch.gather(pred_box_init,1, pred_best_ind[:,None].repeat(1,1,4)).squeeze(1)
        pred_box_best = torch.gather(pred_box_refine,1, pred_best_ind[:,None].repeat(1,1,4)).squeeze(1)

        # anchor_box_center = anchor_box[center_ind_batch, center_ind]
        gt_boxes_vec_center = box_xywh_to_xyxy(gt_bbox).view(-1, 4).clamp(min=0.0, max=1.0)


        if self.cfg.TRAIN.TRAIN_CLS:
            gt_q_map = gt_q_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
                                self.search_feat_size,self.search_feat_size)
            # gt_c_map = gt_c_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
            #                     self.search_feat_size,self.search_feat_size)
        

        
        if self.cfg.TRAIN.TRAIN_CLS:
            nonzero_indices=torch.nonzero(gt_dict['label'].view(-1))
            gt_bbox = gt_bbox[nonzero_indices[:,0],:]
            best_boxes = best_boxes[nonzero_indices[:,0],:]


        giou_loss_init_center, iou_init_center = self.objective['giou'](pred_box_init_center, gt_boxes_vec_center)
        giou_loss_refine_center, iou_refine_center = self.objective['giou'](pred_box_refine_center, gt_boxes_vec_center)
        _, iou_pred = self.objective['giou'](pred_box_best, gt_boxes_vec_center)
        l1_loss_init_center = self.objective['l1'](pred_box_init_center, gt_boxes_vec_center) 
        l1_loss_final_center = self.objective['l1'](pred_box_refine_center, gt_boxes_vec_center) 
        giou_loss_refine_pos, _ = self.objective['giou'](pred_box_refine_pos, gt_box_pos)
        l1_loss_refine_pos = self.objective['l1'](pred_box_refine_pos, gt_box_pos) 
         # (BN,4) (BN,4)
        # _, iou_anchor = self.objective['giou'](anchor_box_center, gt_boxes_vec_center)

        # a = box_xyxy_to_xywh(anchor_box_center)
        # b = box_xyxy_to_xywh(gt_boxes_vec_center)
        # Map loss
        qf_loss = self.objective['qfl'](pred_cls_score_vec, gaussian_center_map.reshape(-1), gaussian_center_map.reshape(-1))
        # qf_loss = self.objective['focal'](pred_cls_score_vec.sigmoid(), gaussian_center_map.reshape(-1,1))
        qf_loss = qf_loss.reshape(self.bs,-1).mean()
        # DFL only handle the selected points inside the box
        # df_loss = self.objective['dfl'](pred_ltrb_pos, target_ltrb_pos, weight=target_weight[:, None].expand(-1, 4).reshape(-1))
        # giou_loss = (giou_loss * target_weight).mean() # weighted GIoU
        giou_loss_init_center = giou_loss_init_center.mean()
        giou_loss_refine_center = giou_loss_refine_center.mean()
        giou_loss_refine_pos = giou_loss_refine_pos.mean()
                
        # weighted sum
        loss = giou_loss_init_center * 2\
        + giou_loss_refine_center * 2\
        + qf_loss\
        + l1_loss_init_center * 5\
        + l1_loss_final_center * 5\
        + giou_loss_refine_pos  * 2\
        + l1_loss_refine_pos
            # + df_loss * 0.25\
            # + qs_loss * 0.25
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = iou_pred.detach().mean()
        mean_iou_init = iou_init_center.detach().mean()
        mean_giou_loss = giou_loss_refine_center.detach()
        
        status = {"Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/l1": l1_loss_final_center.item(),
                #   "Loss/ctr": qs_loss.item(),
                    # "Loss/dfl": df_loss.item(),
                    "Loss/qfl": qf_loss.item(),
                    "IoU": mean_iou.item(),
                    "IoU_init": mean_iou_init.item(),
                    # "IoU_anchor": iou_anchor.mean().item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status
    
    def compute_losses_DECODER(self, pred_dict, gt_dict):
        # K = 60
        # num_classes = 1
        no_init_ce = False
        iou_aware_ce = True
        outputs_without_aux = {
            k: v for k, v in pred_dict.items() if k != "aux_outputs" and k != "enc_outputs"
        }
        # Get init(enc) and final
        pred_query_ind = outputs_without_aux['query_index']
        # pred_query_attn = outputs_without_aux['query_attn']
        src_logits = outputs_without_aux['pred_logits']
        _, K, num_classes = src_logits.shape
        pred_box_init_vec = outputs_without_aux['init_pred_boxes'].view(-1,4)      # xywh
        pred_box_final_vec = outputs_without_aux['pred_boxes'].reshape(-1,4)      # xywh
        pred_logit_init = outputs_without_aux['init_pred_logits']
        pred_logit_final = outputs_without_aux['pred_logits']
        pred_init_all_logits = outputs_without_aux['all_init_logits']
        pred_logit_init_vec = pred_logit_init.reshape(-1, num_classes)
        pred_logit_final_vec = pred_logit_final.reshape(-1, num_classes) 
        # all_init_logits = outputs_without_aux['all_init_logits'].view(self.bs, self.search_feat_size, self.search_feat_size)


        # GT
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_box_vec = gt_bbox.unsqueeze(1).repeat(1,K,1).view(-1,4)

        gt_box_vec_xyxy = box_xywh_to_xyxy(gt_box_vec).clamp(min=0.0, max=1.0)
        
        iou_matrix_init, _ = box_iou(gt_box_vec_xyxy, box_cxcywh_to_xyxy(pred_box_init_vec))
        iou_matrix_init_ = iou_matrix_init.reshape(self.bs, -1)
        iou_matrix_final, _ = box_iou(gt_box_vec_xyxy, box_cxcywh_to_xyxy(pred_box_final_vec))
        iou_matrix_final_ = iou_matrix_final.reshape(self.bs, -1)
        # filtered_pred_init, filtered_gt_init, box_pos_index_init = filter_predictions_by_iou(gt_bbox, outputs_without_aux['init_pred_boxes'], iou_threshold=0.7, maxk=2)
        # filtered_pred_init, filtered_gt_final, box_pos_index_final = filter_predictions_by_iou(gt_bbox, outputs_without_aux['init_pred_boxes'], iou_threshold=0.7, maxk=2)
        # _, _, cls_pos_index_init = filter_predictions_by_iou(gt_bbox, outputs_without_aux['init_pred_boxes'], iou_threshold=0.7, maxk=K, dynamic=False)
        _, _, cls_pos_index_final = filter_predictions_by_iou(gt_bbox, outputs_without_aux['init_pred_boxes'], iou_threshold=0.7, maxk=5, dynamic=False)
        # _, _, cls_nonneg_index = filter_predictions_by_iou(gt_bbox, outputs_without_aux['pred_boxes'], iou_threshold=0.5, maxk=K, dynamic=False)
        # box_pos_index = torch.arange()
        # CDN
        dn_meta = pred_dict['dn_meta']
        if dn_meta is not None and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_padding, dn_num_group = prep_for_dn(dn_meta)
            scalar = dn_num_group
            dn_pred_box = output_known_lbs_bboxes['pred_boxes']
            dn_pred_logits = output_known_lbs_bboxes['pred_logits']
            dn_gt_box = gt_bbox.unsqueeze(1).repeat(1,dn_num_group*single_padding,1).view(-1,4)
            dn_iou, _ = box_iou(box_xywh_to_xyxy(dn_gt_box).clamp(min=0.0, max=1.0), box_cxcywh_to_xyxy(dn_pred_box.reshape(-1,4)))
            # fetch the 1st box of each dn group
            dn_pred_box_positive = dn_pred_box.unsqueeze(1).reshape(self.bs, dn_num_group, single_padding, 4)[:, :, 0].reshape(-1,4)
            dn_gt_box_pos = gt_bbox.unsqueeze(1).repeat(1,dn_num_group,1).view(-1,4)
            dn_gt_labels = torch.linspace(1,0,2)[None].repeat(self.bs, dn_num_group).cuda()
            # iou_cdn = 
                                           
        if torch.isnan(pred_box_init_vec).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        query_map = torch.zeros([self.bs, self.search_feat_size * self.search_feat_size], device=pred_logit_final_vec.device)
        query_map_ = query_map.clone()
        query_map.scatter_(1, pred_query_ind, 1)
        # query_map_.scatter_(1, pred_query_attn, 1)
        query_map_ = query_map_.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        
        gt_box_map =  generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.search_feat_size)

        gt_inbox_labels = pred_logit_final_vec.new_zeros(pred_logit_final_vec.shape)
        gt_cls_labels_init = pred_logit_final_vec.new_zeros(pred_logit_final_vec.shape)
        gt_cls_labels_final = pred_logit_final_vec.new_zeros(pred_logit_final_vec.shape)
        new_gt_cls_labels_final = pred_logit_final_vec.new_zeros(pred_logit_final_vec.shape) -1
        # gt_cls_labels_final = pred_logit_final_vec.new_zeros(pred_logit_final_vec.shape)
        # gt_cls_labels = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # gt_labels[box_pos_index] = 1
        new_gt_cls_labels_final.view(-1)[iou_matrix_final<0.5] = 0
        # new_gt_cls_labels_final[cls_pos_index_final] = 1
        supervision_idx = (new_gt_cls_labels_final>=0)
        new_gt_cls_labels_final = new_gt_cls_labels_final[supervision_idx]
        # gt_cls_labels_init[cls_pos_index_init] = 1
        # gt_cls_labels_final[cls_pos_index_final] = 1
        gt_cls_labels_final[cls_pos_index_final] = 1
        # pred_box_init_pos =  box_cxcywh_to_xyxy(pred_box_init_vec[box_pos_index]).clamp(min=0.0, max=1.0)
        # pred_box_final_pos =  box_cxcywh_to_xyxy(pred_box_final_vec[box_pos_index]).clamp(min=0.0, max=1.0)
        # gt_box_pos = box_xywh_to_xyxy(filtered_gt_init).clamp(min=0.0, max=1.0)

        

        # status for log
        scores = outputs_without_aux['pred_logits'].squeeze(dim=2) 
        top_scores, top_indices = torch.max(scores, dim=1)
        pred_top_ind = torch.gather(pred_query_ind,1,top_indices[:,None])
        pred_top_map = torch.zeros_like(query_map).scatter_(1, pred_top_ind, 1)
        iou_map = torch.zeros_like(query_map).scatter_(1, pred_query_ind, iou_matrix_final.reshape(self.bs, -1))
        init_iou_map = torch.zeros_like(query_map).scatter_(1, pred_query_ind, iou_matrix_init.reshape(self.bs, -1))
        pred_cls_map_init = torch.zeros_like(query_map, dtype=torch.float16).scatter_(1, pred_query_ind, pred_logit_init.squeeze(-1))
        pred_cls_map_final = torch.zeros_like(query_map, dtype=torch.float16).scatter_(1, pred_query_ind, pred_logit_final.squeeze(-1).sigmoid())
        # pred_attn_map = torch.zeros_like(query_map).scatter_(1, pred_query_ind, pred_query_attn)
        # topk_iou_map = torch.zeros_like(query_map, dtype=torch.float16).scatter_(1, pred_query_ind, gt_cls_labels.reshape(self.bs, -1))
        # label_cross_map = pred_attn_map * gt_box_map.reshape(self.bs,-1)
        # targets = torch.gather(gt_box_map.flatten(1), 1, pred_query_ind)

        # DEBUG maps
        pred_top_map = pred_top_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        query_map = query_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        iou_map = iou_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        init_iou_map = init_iou_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        # topk_iou_map = topk_iou_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        pred_cls_map_init = pred_cls_map_init.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        pred_cls_map_final = pred_cls_map_final.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        gaussian_center_map = get_2d_gaussian_map(xywh_to_cxcywh(gt_bbox), self.search_feat_size, alpha=0.5)
        norm_center_map = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)[0]
        inbox_topk_map = query_map * gt_box_map
        box_targets = torch.gather(inbox_topk_map.flatten(1), 1, pred_query_ind)
        gaussian_center_targets = torch.gather(norm_center_map.flatten(1), 1, pred_query_ind)
        q_in_box_ind = box_targets.reshape(-1).nonzero().reshape(-1)

        num_cls_samples = reduce_mean(
            torch.tensor(box_targets.sum()).cuda()).item()
        # num_cls_samples = 1
        topK = 4
        num_box_samples = reduce_mean(
            torch.tensor(self.bs*topK, dtype=torch.float32).cuda()).item()
        if self.count == 12:
            print("###############num_box_samples##############")
            print(num_box_samples)
        gt_inbox_labels[q_in_box_ind] = 1
        
        inbox_gt_box = gt_box_vec_xyxy[gt_inbox_labels.squeeze(1) > 0]
        inbox_pred_box_init = box_cxcywh_to_xyxy(pred_box_init_vec[gt_inbox_labels.squeeze(1) > 0])
        inbox_pred_box_final = box_cxcywh_to_xyxy(pred_box_final_vec[gt_inbox_labels.squeeze(1) > 0])
        # topN_and_inbox_bit =  torch.isin(torch.where(gt_inbox_labels.squeeze(1) > 0)[0], box_pos_index_final.cuda())
        # inbox_pred_box_init = inbox_pred_box_init[topN_and_inbox_bit]
        # inbox_pred_box_final = inbox_pred_box_final[topN_and_inbox_bit]
        # inbox_gt_box = inbox_gt_box[topN_and_inbox_bit]
        # gt_cls_labels[cls_pos_index] = 1

        # Every first query as positive box
        
        pred_box_init_xyxy = box_cxcywh_to_xyxy(outputs_without_aux['init_pred_boxes'])
        pred_box_final_xyxy = box_cxcywh_to_xyxy(outputs_without_aux['pred_boxes'])
        pred_box_final_pos = torch.gather(pred_box_final_xyxy,1,pred_logit_final[...,0].topk(topK)[1][...,None].expand(-1,-1,4)).view(-1,4)
        pred_box_init_pos = torch.gather(pred_box_init_xyxy,1,pred_logit_final[...,0].topk(topK)[1][...,None].expand(-1,-1,4)).view(-1,4)
        gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)
        gt_box_pos = gt_box_init_pos = gt_bbox_xyxy[:,None].repeat(1,topK,1).view(-1, 4)

        # gt_box_pos = box_xywh_to_xyxy(filtered_gt_final).clamp(min=0.0, max=1.0)
        # gt_box_init_pos = box_xywh_to_xyxy(filtered_gt_init).clamp(min=0.0, max=1.0)
        # pred_box_init_pos =  box_cxcywh_to_xyxy(pred_box_init_vec[box_pos_index_init])
        # pred_box_final_pos =  box_cxcywh_to_xyxy(pred_box_final_vec[box_pos_index_final])
        # gt_box_pos = box_xywh_to_xyxy(filtered_gt_init).clamp(min=0.0, max=1.0)

        ############################
        #  Simple Hungarian Match  #
        ############################
        # box_giou_cost = self.objective['giou'](gt_box_vec_xyxy, box_cxcywh_to_xyxy(pred_box_final_vec))[0]-1 # bs,N
        # box_l1_cost = torch.pairwise_distance(gt_box_vec_xyxy, box_cxcywh_to_xyxy(pred_box_final_vec), p=1)
        # out_prob = pred_logit_final_vec.sigmoid()
        # alpha = 0.25
        # gamma = 2.0
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-5).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-5).log())
        # cost_class = pos_cost_class[:, 0] - neg_cost_class[:, 0]
        # Cost = 2.0*cost_class + 5.0*box_l1_cost + 2.0*box_giou_cost
        # min_ind = Cost.view(self.bs,-1).min(1,keepdim=True)[1] # bs, 1
        # cls_targets = src_logits.new_zeros(src_logits.shape).squeeze(-1)
        # cls_targets.scatter_(1, min_ind, 1)
        ############################
        #           End            #
        ############################
        
        # pred_box_init_pos = torch.gather(pred_box_init_xyxy,1, min_ind[...,None].expand(-1,-1,4)).view(-1,4)
        # pred_box_final_pos = torch.gather(pred_box_final_xyxy,1, min_ind[...,None].expand(-1,-1,4)).view(-1,4)


        
        # ce_loss_init = self.objective['qfl'](inverse_sigmoid(pred_logit_init_vec), gt_box_map.reshape(-1), iou_matrix_init.detach())
        # LOSS
        if iou_aware_ce:
            # ce_loss_init = self.objective['dfocal'](inverse_sigmoid(pred_logit_init_vec) , iou_matrix_init[:,None].detach())
            if no_init_ce:
                ce_loss_init = 0
            else:
                ce_loss_init = self.objective['qfl'](pred_logit_init_vec, box_targets.reshape(-1), iou_matrix_init.detach())
            # ce_loss_final = self.objective['qfl'](pred_logit_final_vec, box_targets.reshape(-1), iou_matrix_final.detach())
            # ce_loss_final = self.objective['focal'](pred_logit_final_vec.sigmoid().view(self.bs,-1), gaussian_center_targets)
            ce_loss_final = self.objective['qfl'](pred_logit_final_vec, gt_inbox_labels.reshape(-1), iou_matrix_init.detach())
            # ce_loss_init = self.objective['qfl'](inverse_sigmoid(pred_logit_init_vec), gt_inbox_labels.reshape(-1), iou_matrix_init.detach())
            # ce_loss_final = self.objective['dfocal'](pred_logit_final_vec, gt_cls_labels_final)
        else:
            # ce_loss_init = self.objective['dfocal'](pred_logit_init, gt_cls_labels_init.reshape(self.bs,-1, num_classes))
            # ce_loss_final = self.objective['dfocal'](pred_logit_final, gt_cls_labels_final.reshape(self.bs,-1, num_classes))
            # ce_loss_init = self.objective['dfocal'](pred_logit_init_vec, gt_cls_labels_init)
            ce_loss_init = 0
            # pred_logit_final_vec = pred_logit_final_vec[supervision_idx]
            # ce_loss_final = self.objective['dfocal']((pred_logit_final_vec), gaussian_center_map.flatten(1)[...,None])
            # ce_loss_final = self.objective['dfocal']((pred_logit_final_vec), cls_targets.view(-1,1))
            # ce_loss_init = self.objective['qfl'](pred_logit_init_vec, box_targets.reshape(-1), iou_matrix_init.detach())
            ce_loss_final = self.objective['qfl'](pred_logit_final_vec, box_targets.reshape(-1), iou_matrix_final.detach())
            # ce_loss_final = self.objective['qfl'](pred_logit_final_vec, gt_inbox_labels.reshape(-1), iou_matrix_final.detach())
            # ce_loss_init = self.objective['focal'](pred_init_all_logits.sigmoid().view(self.bs,-1), gaussian_center_targets)
            # ce_loss_final = self.objective['focal'](pred_logit_final_vec.sigmoid().view(self.bs,-1), gaussian_center_targets)
        
        # ce_loss_final = self.objective['qfl'](pred_logit_final_vec, box_targets.reshape(-1), iou_matrix_final)
        giou_loss_init, iou_init = self.objective['giou'](pred_box_init_pos, gt_box_init_pos)
        giou_loss_final, iou_final = self.objective['giou'](pred_box_final_pos, gt_box_pos)
        l1_loss_init = self.objective['l1'](pred_box_init_pos, gt_box_init_pos)  # (BN,4) (BN,4)
        l1_loss_final = self.objective['l1'](pred_box_final_pos, gt_box_pos)  # (BN,4) (BN,4)
        # giou_loss_init, iou_init = self.objective['giou'](inbox_pred_box_init, inbox_gt_box)
        # giou_loss_final, iou_final = self.objective['giou'](inbox_pred_box_final, inbox_gt_box)
        # l1_loss_init = self.objective['l1'](inbox_pred_box_init, inbox_gt_box)  # (BN,4) (BN,4)
        # l1_loss_final = self.objective['l1'](inbox_pred_box_final, inbox_gt_box)  # (BN,4) (BN,4)

        # CDN LOSS
        if dn_meta is not None:
            dn_pred_box_pos =  box_cxcywh_to_xyxy(dn_pred_box_positive)
            dn_gt_box_pos = box_xywh_to_xyxy(dn_gt_box_pos).clamp(min=0.0, max=1.0)
            giou_loss_cdn, iou_cdn_pos = self.objective['giou'](dn_pred_box_pos, dn_gt_box_pos)
            if iou_aware_ce:
                # ce_loss_cdn = self.objective['dfocal'](dn_pred_logits.reshape(-1,1), dn_iou.view(-1,1).detach())
                # ce_loss_cdn = self.objective['focal'](dn_pred_logits.sigmoid().view(self.bs,-1), dn_gt_labels)
                ce_loss_cdn = self.objective['qfl'](dn_pred_logits.reshape(-1,1), dn_gt_labels.reshape(-1), dn_iou.detach())
            else:
                ce_loss_cdn = self.objective['qfl'](dn_pred_logits.reshape(-1,1), dn_gt_labels.view(-1), dn_iou.detach())
                # ce_loss_cdn = self.objective['dfocal'](dn_pred_logits.reshape(-1,1), dn_gt_labels.view(-1,1))
            # ce_loss_cdn = ce_loss_cdn.mean(1).reshape(self.bs,dn_num_group*single_padding).mean(1).sum()
            ce_loss_cdn = ce_loss_cdn.sum() / num_cls_samples / scalar
            
            l1_loss_cdn = self.objective['l1'](dn_pred_box_pos, dn_gt_box_pos, reduction='none')  # (BN,4) (BN,4)
            loss_cdn = ce_loss_cdn * 1.0 + 2.0 * giou_loss_cdn.sum() / num_box_samples / scalar + 5.0 * l1_loss_cdn.sum() / num_box_samples / scalar
            aux_loss_cdn = 0
            for aux_output in output_known_lbs_bboxes['aux_outputs']:
                _dn_pred_box = aux_output['pred_boxes']
                _boxes_vec = _dn_pred_box.unsqueeze(1).reshape(self.bs, dn_num_group, single_padding, 4)[:, :, 0].reshape(-1,4)
                _logits_vec = aux_output['pred_logits'].reshape(-1, num_classes)
                _boxes_pos = box_cxcywh_to_xyxy(_boxes_vec)
                _giou_loss, _iou = self.objective['giou'](_boxes_pos, dn_gt_box_pos)
                _l1_loss = self.objective['l1'](_boxes_pos, dn_gt_box_pos, reduction='none')  # (BN,4) (BN,4)
                _dn_iou, _ = box_iou(box_xywh_to_xyxy(dn_gt_box).clamp(min=0.0, max=1.0), box_cxcywh_to_xyxy(_dn_pred_box.reshape(-1,4)))
                if iou_aware_ce:
                    _ce_loss = self.objective['qfl'](_logits_vec.reshape(-1,1), dn_gt_labels.reshape(-1), _dn_iou.detach())
                    # _ce_loss = self.objective['focal'](_logits_vec.sigmoid().view(self.bs,-1), dn_gt_labels)
                    # _ce_loss = self.objective['dfocal'](_logits_vec,  _dn_iou.view(-1,1).detach())
                else:
                    _ce_loss = self.objective['qfl'](_logits_vec.reshape(-1,1), dn_gt_labels.reshape(-1), _dn_iou.detach())
                    # _ce_loss = self.objective['dfocal'](_logits_vec, dn_gt_labels.view(-1,1))
                # _ce_loss_cdn = _ce_loss.mean(1).reshape(self.bs,dn_num_group*single_padding).mean(1).sum()
                _ce_loss = _ce_loss.sum() / num_cls_samples / scalar
                _l1_loss = self.objective['l1'](_boxes_pos, dn_gt_box_pos)  # (BN,4) (BN,4)
                aux_loss_cdn += 1.0 * _ce_loss + 2.0 * _giou_loss.sum() / num_box_samples / scalar + 5.0 *_l1_loss.sum() / num_box_samples / scalar
        else:
            loss_cdn = torch.as_tensor(0.0).to("cuda")
            aux_loss_cdn = torch.as_tensor(0.0).to("cuda")

        # AUX LOSS
        aux_loss = 0
        for aux_output in pred_dict['aux_outputs']:
            _logits = aux_output['pred_logits']
            _logits_vec = _logits.reshape(-1, num_classes)
            _boxes_vec = aux_output['pred_boxes']#.reshape(-1,4)
            # _boxes_pos = box_cxcywh_to_xyxy(_boxes_vec[box_pos_index_final])
            _boxes_pos = torch.gather(_boxes_vec,1,_logits[...,0].topk(topK)[1][...,None].expand(-1,-1,4)).view(-1,4)
            _boxes_pos = box_cxcywh_to_xyxy(_boxes_pos)
            _giou_loss, _ = self.objective['giou'](_boxes_pos, gt_box_pos)
            _iou, _ = box_iou(box_xywh_to_xyxy(gt_box_vec).clamp(min=0.0, max=1.0), box_cxcywh_to_xyxy(_boxes_vec.reshape(-1,4)))
            if iou_aware_ce:
                # _ce_loss = self.objective['dfocal'](inverse_sigmoid(_logits_vec) , _iou[:,None].detach())
                _ce_loss = self.objective['qfl'](_logits_vec, box_targets.reshape(-1), _iou.detach())
                # _ce_loss = self.objective['focal'](_logits_vec.sigmoid().view(self.bs,-1), gaussian_center_targets)
                # _ce_loss = self.objective['dfocal'](_logits_vec, gt_cls_labels_final)
                # _ce_loss = self.objective['qfl'](_logits_vec, gt_inbox_labels.reshape(-1), _iou.detach())
            else:
                # _ce_loss = self.objective['dfocal'](_logits_vec, cls_targets.view(-1,1))
                _ce_loss = self.objective['qfl'](_logits_vec, box_targets.reshape(-1), _iou.detach())
                # _ce_loss = self.objective['dfocal'](_logits, gt_cls_labels_final.reshape(self.bs,-1, num_classes))
                # _ce_loss = self.objective['dfocal'](_logits_vec[supervision_idx,], new_gt_cls_labels_final)
                # _ce_loss = self.objective['focal'](_logits_vec.sigmoid().view(self.bs,-1), gaussian_center_targets)
            # _ce_loss = _ce_loss.mean(1).reshape(self.bs,K).sum(1).mean()
            _ce_loss = _ce_loss.sum() / num_cls_samples
            # _ce_loss = 0
            _l1_loss = self.objective['l1'](_boxes_pos, gt_box_pos, reduction='none')  # (BN,4) (BN,4)
            aux_loss += 1.0 * _ce_loss + 2.0 * _giou_loss.sum() / num_box_samples + 5.0 *_l1_loss.sum() / num_box_samples

        # giou_loss = (giou_loss * target_weight).mean() # weighted GIoU
        # ce_loss_init = ce_loss_init.mean(1).reshape(self.bs,-1).sum(1).mean()
        # ce_loss_final = ce_loss_final.mean(1).reshape(self.bs,K).sum(1).mean()
        ce_loss_final = ce_loss_final.sum() / num_cls_samples
        if not no_init_ce:
            ce_loss_init = ce_loss_init.sum() / num_cls_samples #.reshape(self.bs,K).mean(1).sum()
        # ce_loss_init = ce_loss_init / K
        giou_loss_init = giou_loss_init.sum() / num_box_samples
        giou_loss_final = giou_loss_final.sum() / num_box_samples
        # save_masked_img(gt_dict, 11,alpha=0.15,boxxywh=outputs_without_aux['pred_boxes'])
        init_box_loss = giou_loss_init * 2.0 + l1_loss_init * 5.0
        final_box_loss = giou_loss_final * 2.0 + l1_loss_final * 5.0
                
        # weighted sum
        loss = 0 +\
        + ce_loss_init * 1.0\
        + ce_loss_final * 1.0\
        + init_box_loss * 1.0\
        + final_box_loss * 1.0\
        + aux_loss * 1.0\
        + loss_cdn * 1.0\
        + aux_loss_cdn * 1.0

        
        
        
        # targets = gt_cls_labels.new_zeros((self.bs, 256).shape)
        # for i, m in enumerate(label_cross_map):
        #     if m.sum()<=0:
        #         max_id = gaussian_center_map[i].view(-1).argmax()
        #     else:
        #         max_id = m.view(-1).argmax()
        #     targets[i, max_id] = 1

        top_boxes = torch.gather(outputs_without_aux['pred_boxes'], 1, top_indices[:,None,None].expand(-1, -1, 4)).squeeze(dim=1) 
        pred_iou, _ = box_iou(box_cxcywh_to_xyxy(top_boxes), box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0))
        query_corre_num = (gt_box_map*query_map).sum(1).sum(1)
        # search_img = gt_dict['search_images'][0][7]*torch.asarray([0.229, 0.224, 0.225], device=scores.device)[:,None,None]+\
        #     torch.asarray([0.485, 0.456, 0.406], device=scores.device)[:,None,None]
        # torchvision.utils.save_image(search_img, 'xxxxxx.png')
        # F.interpolate(gt_box_map[None,:], size=(256,256))
        # torchvision.utils.save_image((gt_dict['search_images'][0][2]*torch.asarray([0.229, 0.224, 0.225], device=scores.device)[:,None,None]+\
        #    torch.asarray([0.485, 0.456, 0.406], device=scores.device)[:,None,None])*F.interpolate(gt_box_map[None,:], size=(256,256))[0,2], 'xxxxxx.png')
        mean_iou_pred = pred_iou.detach().mean()
        # mean_iou_attn = iou_matrix_final_[:,0].mean()
        mean_iou_init = iou_init.detach().mean()
        mean_iou_last = iou_final.detach().mean()
        cls_loss = ce_loss_final.detach()
        mean_giou_loss = giou_loss_final.detach()
        # avg_box_pos_num = len(box_pos_index_init)//self.bs
        # avg_cls_pos_num = len(cls_pos_index_final)//self.bs

        status = {  "Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/cls": cls_loss.item(),
                    "IoU_i": mean_iou_init.item(),
                    "IoU_f": mean_iou_last.item(),
                    # "IoU_a": mean_iou_attn.item(),
                    "IoU": mean_iou_pred.item(),
                    # "Pb":avg_box_pos_num,
                    # "Pc":avg_cls_pos_num
                    }
        if len(pred_dict['aux_outputs'])>0:
            status['Loss/aux'] = aux_loss.item()
        return loss, status
     
# [save_masked_img(gt_dict, i, pred_cls_map_final, xyxy=pred_box_final_pos.view(self.bs,-1,4), f_name=f'xxx{i}.png') for i in range(self.bs)]
# save_masked_img(gt_dict, 0, pred_cls_map_final, xyxy=box_xywh_to_xyxy(gt_bbox[:,None]))
# save_masked_img(gt_dict, 0, pred_cls_map_final, xyxy=pred_box_final_xyxy)
# [save_masked_img(gt_dict, i, pred_cls_map_final, xyxy=box_xywh_to_xyxy(gt_bbox[:,None]), f_name=f'xxx{i}.png') for i in range(self.bs)]
def save_masked_img(gt_dict, batch_id, mask_map=None, src='search', f_name='xxx.png', alpha=.2, xyxy=None):
    image_batch = gt_dict[src+'_images'][0,batch_id].cpu()
    std = torch.asarray([0.229, 0.224, 0.225])[:,None,None]
    mean = torch.asarray([0.485, 0.456, 0.406])[:,None,None]

    shape = gt_dict[src+'_images'].shape[-1]

    if mask_map is not None:
        mask = F.interpolate(mask_map[None,:], size=shape)[0, batch_id].cpu() + alpha
    else:
        mask = 1
    img = (image_batch * std + mean)*mask
    if xyxy is not None:
        img_int = (img*256).clamp(0,255.9).to(torch.uint8)
        boxxyxy = xyxy[batch_id] * shape
        img_int = torchvision.utils.draw_bounding_boxes(img_int, boxxyxy, width=1)
        img = img_int.to(torch.float32)/256.0
    torchvision.utils.save_image(img, f_name)


def filter_scores(S, Ind, bs, alpha=1.0):
    # Compute the batch indices
    batch_indices = Ind[:, 0]

    # Compute the mean and standard error for each batch
    batch_means = torch.zeros(bs, dtype=torch.float32, device=S.device)
    batch_std = torch.zeros(bs, dtype=torch.float32, device=S.device)
    for i in range(bs):
        mask = (Ind[:, 0] == i)
        batch_scores = S[mask]
        batch_size = batch_scores.size(0)
        if batch_size > 0:
            batch_means[i] = batch_scores.mean()
            batch_std[i] = batch_scores.std() / torch.sqrt(torch.tensor(batch_size))

    # Compute the threshold for each batch
    batch_thresholds = batch_means + alpha * batch_std

    # Compute the mask for the scores that pass the threshold
    # mask = S > batch_thresholds[batch_indices]
    mask = S > 0

    # Filter the indices and scores
    filtered_indices = Ind[mask]
    filtered_scores = S[mask]

    return filtered_indices, filtered_scores, mask

   
def batch_box_iou(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [bs, N, 2]
    rb = torch.min(boxes1[:, None, 2:4] + boxes1[:, None, :2], boxes2[:, 2:4] + boxes2[:, :2])  # [bs, N, 2]

    wh = (rb - lt).clamp(min=0)  # [bs, N, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [bs, N]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def filter_predictions_by_iou(GT_boxes, pred_boxes, iou_threshold=0.5, k=1, maxk=5, dynamic=False):
    bs, n = pred_boxes.shape[:2]

    # Calculate IoU between GT_boxes and pred_boxes
    GT_vec = box_xywh_to_xyxy(GT_boxes)[:, None, :].repeat((1, n, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
    P_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4).clamp(min=0.0, max=1.0)
    iou, _ = box_iou(GT_vec, P_vec)
    iou = iou.view(bs,n)

    # Filter predictions based on IoU threshold
    filtered_preds = []
    filtered_gts = []
    positive_indices = []
    for i in range(bs):
        if dynamic:
            iou_threshold = iou[i].mean()+iou[i].std()
        valid_indices = (iou[i] >= iou_threshold).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) >= maxk:
            topk_indices = iou[i].topk(maxk, largest=True, sorted=True).indices
            valid_indices = topk_indices
        # If no predictions meet the condition, reserve the top-k predictions
        if len(valid_indices) == 0:
            topk_indices = iou[i].topk(k, largest=True, sorted=True).indices
            valid_indices = topk_indices

        filtered_preds.append(pred_boxes[i, valid_indices])
        filtered_gts.append(GT_boxes[i].expand(len(valid_indices), -1))

        # Store the indices of positive samples
        positive_indices.extend((i * n + valid_indices).tolist())

    return torch.cat(filtered_preds), torch.cat(filtered_gts), torch.tensor(positive_indices)

def __filter_predictions_by_iou(gt_boxes, pred_boxes, iou_threshold=0.5, k=1):
    bs, n = pred_boxes.shape[:2]

    # Function to compute IoU between two boxes
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        X1, Y1, X2, Y2 = box2

        w_intersection = max(0, min(x2, X2) - max(x1, X1))
        h_intersection = max(0, min(y2, Y2) - max(y1, Y1))
        area_intersection = w_intersection * h_intersection

        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (X2 - X1) * (Y2 - Y1)
        area_union = area_box1 + area_box2 - area_intersection

        return area_intersection / area_union

    filtered_pred_boxes = []
    filtered_gt_boxes = []

    for i in range(bs):
        gt_box = gt_boxes[i]
        preds = pred_boxes[i]

        # Compute IoU values between the GT box and predicted boxes
        iou_values = torch.tensor([iou(gt_box, pred) for pred in preds])

        # Filter predictions by the IoU threshold
        filtered_preds = preds[iou_values >= iou_threshold]

        # If no predictions meet the threshold, keep the top-k predictions based on the IoU values
        if len(filtered_preds) == 0:
            topk_indices = iou_values.topk(k, dim=0)[1]
            filtered_preds = preds[topk_indices]

        filtered_pred_boxes.append(filtered_preds)
        filtered_gt_boxes.append(gt_box.unsqueeze(0).repeat(len(filtered_preds),1))

    return filtered_pred_boxes, filtered_gt_boxes

def prep_for_dn(dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups