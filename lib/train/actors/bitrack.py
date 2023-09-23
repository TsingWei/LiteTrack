from . import BaseActor
from lib.utils.misc import NestedTensor, inverse_sigmoid
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou, box_iou_pairwise, xywh_to_cxcywh
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap, generate_cls_map, generate_distribution_heatmap, grid_center_2d, grid_center_flattened, bbox2distance, get_2d_gaussian_map
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F


class BiTrackActor(BaseActor):
    """ Actor for training BiTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.search_feat_size = self.cfg.DATA.SEARCH.SIZE//self.cfg.MODEL.BACKBONE.STRIDE
        self.template_feat_size = self.cfg.DATA.TEMPLATE.SIZE//self.cfg.MODEL.BACKBONE.STRIDE

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
        # mock
        template_bb = data['template_anno'][0]
        search_bb = data['search_anno'][0]
        # data['rand_template_bb'] = jitter_box_in_box(template_bb,5)
        # data['rand_search_bb'] = jitter_box_in_box(search_bb,5)
        
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        template_target_mask = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                             data['template_anno'][0], generate_bb_mask_only=False)
        # if self.cfg.MODEL.BACKBONE.CE_LOC:
            # ce_box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
            #                                 data['template_anno'][0])

        #     ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
        #     ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
        #     ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
        #                                         total_epochs=ce_start_epoch + ce_warm_epoch,
        #                                         ITERS_PER_EPOCH=1,
        #                                         base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        template_bb = data['template_anno'][0]
        template_bb = box_xywh_to_xyxy(template_bb).clamp(min=0.0,max=1.0)
        search_bb = data['search_anno'][0].clamp(min=0.0,max=1.0)
        search_bb = box_xywh_to_xyxy(search_bb).clamp(min=0.0,max=1.0)
        # rand_template_bb = data['rand_template_bb']
        # rand_search_bb = data['rand_search_bb']
        # template_target_mask = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
        #                                      data['template_anno'][0], generate_bb_mask_only=True)
        # if self.cfg.MODEL.USE_CLS_TOKEN:
        # gt_gaussian_maps_T = generate_heatmap(data['search_anno'], 16)
        # gt_gaussian_maps_T = gt_gaussian_maps_T[-1].unsqueeze(1)
        gt_cls_maps = generate_cls_map(box_xywh_to_xyxy(data['search_anno'][-1]), self.search_feat_size)
        gt_cls_maps_T = generate_cls_map(box_xywh_to_xyxy(data['template_anno'][-1]), self.template_feat_size)

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            return_last_attn=False,
                            template_bb=template_bb,
                            search_box=search_bb,
                            gt_score_map=gt_cls_maps,
                            gt_score_map_T=gt_cls_maps_T,
                            mask_z=template_target_mask,
                            # mock_template_bb=rand_template_bb,
                            # mock_search_bb=rand_search_bb,
                            )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if self.cfg.MODEL.HEAD.TYPE == 'GFL' or self.cfg.MODEL.HEAD.TYPE == 'PyGFL':
            loss, status = self.compute_losses_GFL(pred_dict, gt_dict,)
            loss_T, status_T = self.compute_losses_GFL_T(pred_dict, gt_dict,)
            loss = loss+0.5*loss_T
            status.update(status_T)
        if self.cfg.MODEL.HEAD.TYPE == 'MLP':
            loss, status = self.compute_losses_MLP(pred_dict, gt_dict,)
            # loss_T, status_T = self.compute_losses_GFL_T(pred_dict, gt_dict,)
            # status.update(status_T)
        if return_status:
            return loss, status
        else:
            return loss

    def compute_losses_GFL(self, pred_dict, gt_dict):
        # Get GT boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # Get pred
        pred_boxes = pred_dict['pred_boxes']
        # pred_center_score_unsig = pred_dict['score_map_c']
        pred_quality_score = pred_dict['score_map_q']
        best_boxes = pred_dict['box_best']
        
        # c_score_unsig = pred_center_score_unsig.permute(0, 2, 3,
        #                               1).reshape(-1, 256)
        q_score = pred_quality_score.permute(0, 2, 3,
                                      1).reshape(-1, 1)
                                           
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        
        gt_q_map = generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.search_feat_size)
        gt_c_map = generate_heatmap(gt_dict['search_anno'], self.search_feat_size)[0]

        # if self.cfg.TRAIN.TRAIN_CLS:
        #     gt_q_map = gt_q_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
        #                         self.search_feat_size,self.search_feat_size)
        #     gt_c_map = gt_c_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
        #                         self.search_feat_size,self.search_feat_size)
        
        # Duplicate GT box
        bs = gt_q_map.shape[0]
        inside_ind = torch.nonzero(gt_q_map.flatten(1))
        gt_bboxes_multiple = gt_bbox[inside_ind[:,0],:]
        gt_boxes_vec = box_xywh_to_xyxy(gt_bboxes_multiple).clamp(min=0.0,max=1.0)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)

        iou, _ = box_iou(pred_boxes_vec, gt_boxes_vec)
        iou = iou.detach()
        q_score_pos = pred_quality_score.squeeze(1).flatten(1)[inside_ind[:,0], inside_ind[:,0][1]]

        # # Adaptive select the pos sample
        # pos_ind, selected_iou, select_mask = filter_scores(iou, inside_ind, bs=bs) # ([bs_ind, flatten_map_ind])
        # q_score_pos_unsig = q_score_unsig.flatten(1)[pos_ind[:,0], pos_ind[:,0][1]]
        # pred_box_pos = pred_boxes_vec[select_mask]
        # target_box_pos = gt_boxes_vec[select_mask]
        
        # if self.cfg.TRAIN.TRAIN_CLS:
        #     nonzero_indices=torch.nonzero(gt_dict['label'].view(-1))
        #     gt_bbox = gt_bbox[nonzero_indices[:,0],:]
        #     best_boxes = best_boxes[nonzero_indices[:,0],:]
        top_gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0)
        top_pred_boxes_vec = best_boxes

        giou_loss, _ = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        # giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        top_giou, top_iou = self.objective['giou'](top_pred_boxes_vec, top_gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # Map loss
        iou_score = gt_q_map.new_zeros(gt_q_map.flatten(1).shape) 
        iou_score[inside_ind[:,0], inside_ind[:, 1]] = iou

        # Only fetch selected samples.
        pred_ltrb = pred_dict['ltrb_dis'] # already inside the GT box
        pred_ltrb_pos = pred_ltrb.reshape(-1, 17)
        target_ltrb_pos =  bbox2distance(grid_center_2d(gt_q_map, self.search_feat_size),
            gt_boxes_vec * self.search_feat_size, max_dis=self.search_feat_size).reshape(-1)# dont norm to 1

        # The weight for localization, from predicted cls socre.
        target_weight = q_score_pos.detach()

        # Map for classification.
        gaussian_center_map = get_2d_gaussian_map(xywh_to_cxcywh(gt_dict['search_anno'][0]), self.search_feat_size)
        iou_gaussian = iou_score * gaussian_center_map.flatten(1)
        # target_gaussian_weight = gaussian_center_map.flatten(1)[inside_ind[:,0],inside_ind[:,1]] * target_weight
        # gt_gaussian_maps[-1].unsqueeze(1)

        # Focal loss
        # ctr_loss = self.objective['focal'](c_score_unsig.sigmoid().reshape(bs, -1), gt_c_map.reshape(bs, -1))
        # QFL handle all the points inside the box.
        qf_loss = self.objective['qfl'](inverse_sigmoid(q_score), gt_q_map.reshape(-1), iou_gaussian.reshape(-1))
        # DFL only handle the selected points inside the box
        df_loss = self.objective['dfl'](pred_ltrb_pos, target_ltrb_pos, weight=target_weight[:, None].expand(-1, 4).reshape(-1))
        giou_loss = (giou_loss * target_weight).mean() # weighted GIoU
                
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss\
            + self.loss_weight['l1'] * l1_loss\
            + qf_loss\
            + df_loss * 0.25\
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = top_iou.detach().mean()
        mean_giou_loss = top_giou.detach().mean()
        
        status = {"Loss/total": loss.item(),
                    "Loss/giou": mean_giou_loss.item(),
                    "Loss/l1": l1_loss.item(),
                #   "Loss/ctr": ctr_loss.item(),
                    "Loss/dfl": df_loss.item(),
                    "Loss/qfl": qf_loss.item(),
                    "IoU": mean_iou.item(),
                #   "IoU_T": mean_iou.item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status

    def compute_losses_GFL_T(self, pred_dict, gt_dict):
        # Get GT boxes
        gt_bbox = gt_dict['template_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # Get pred
        pred_boxes = pred_dict['pred_boxes_T']
        # pred_center_score_unsig = pred_dict['score_map_c']
        pred_quality_score = pred_dict['score_map_q_T']
        best_boxes = pred_dict['box_best_T']
        
        # c_score_unsig = pred_center_score_unsig.permute(0, 2, 3,
        #                               1).reshape(-1, 256)
        q_score = pred_quality_score.permute(0, 2, 3,
                                      1).reshape(-1, 1)
                                           
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        
        gt_q_map = generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.template_feat_size)
        gt_c_map = generate_heatmap(gt_dict['template_anno'], self.template_feat_size)[0]

        # if self.cfg.TRAIN.TRAIN_CLS:
        #     gt_q_map = gt_q_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
        #                         self.search_feat_size,self.search_feat_size)
        #     gt_c_map = gt_c_map * gt_dict['label'].view(-1)[:,None,None].repeat(1,
        #                         self.search_feat_size,self.search_feat_size)
        
        # Duplicate GT box
        bs = gt_q_map.shape[0]
        inside_ind = torch.nonzero(gt_q_map.flatten(1))
        gt_bboxes_multiple = gt_bbox[inside_ind[:,0],:]
        gt_boxes_vec = box_xywh_to_xyxy(gt_bboxes_multiple).clamp(min=0.0,max=1.0)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)

        iou, _ = box_iou(pred_boxes_vec, gt_boxes_vec)
        iou = iou.detach()
        q_score_pos = pred_quality_score.squeeze(1).flatten(1)[inside_ind[:,0], inside_ind[:,0][1]]

        # # Adaptive select the pos sample
        # pos_ind, selected_iou, select_mask = filter_scores(iou, inside_ind, bs=bs) # ([bs_ind, flatten_map_ind])
        # q_score_pos_unsig = q_score_unsig.flatten(1)[pos_ind[:,0], pos_ind[:,0][1]]
        # pred_box_pos = pred_boxes_vec[select_mask]
        # target_box_pos = gt_boxes_vec[select_mask]
        
        # if self.cfg.TRAIN.TRAIN_CLS:
        #     nonzero_indices=torch.nonzero(gt_dict['label'].view(-1))
        #     gt_bbox = gt_bbox[nonzero_indices[:,0],:]
        #     best_boxes = best_boxes[nonzero_indices[:,0],:]
        top_gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0,max=1.0)
        top_pred_boxes_vec = best_boxes

        giou_loss, _ = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        # giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        top_giou, top_iou = self.objective['giou'](top_pred_boxes_vec, top_gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # Map loss
        iou_score = gt_q_map.new_zeros(gt_q_map.flatten(1).shape) 
        iou_score[inside_ind[:,0], inside_ind[:, 1]] = iou

        # Only fetch selected samples.
        pred_ltrb = pred_dict['ltrb_dis_T'] # already inside the GT box
        pred_ltrb_pos = pred_ltrb.reshape(-1, 17)
        target_ltrb_pos =  bbox2distance(grid_center_2d(gt_q_map, self.template_feat_size),
            gt_boxes_vec * self.template_feat_size, max_dis=self.template_feat_size).reshape(-1)# dont norm to 1

        # The weight for localization, from predicted cls socre.
        target_weight = q_score_pos.detach()

        # Map for classification.
        gaussian_center_map = get_2d_gaussian_map(xywh_to_cxcywh(gt_dict['template_anno'][0]), self.template_feat_size)
        iou_gaussian = iou_score * gaussian_center_map.flatten(1)
        # target_gaussian_weight = gaussian_center_map.flatten(1)[inside_ind[:,0],inside_ind[:,1]] * target_weight
        # gt_gaussian_maps[-1].unsqueeze(1)

        # Focal loss
        # ctr_loss = self.objective['focal'](c_score_unsig.sigmoid().reshape(bs, -1), gt_c_map.reshape(bs, -1))
        # QFL handle all the points inside the box.
        qf_loss = self.objective['qfl'](inverse_sigmoid(q_score), gt_q_map.reshape(-1), iou_gaussian.reshape(-1))
        # DFL only handle the selected points inside the box
        df_loss = self.objective['dfl'](pred_ltrb_pos, target_ltrb_pos, weight=target_weight[:, None].expand(-1, 4).reshape(-1))
        giou_loss = (giou_loss * target_weight).mean() # weighted GIoU
                
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss\
            + self.loss_weight['l1'] * l1_loss\
            + qf_loss\
            + df_loss * 0.25\
            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = top_iou.detach().mean()
        mean_giou_loss = top_giou.detach().mean()
        
        status = {
                    "IoU_T": mean_iou.item(),
                #   "IoU_T": mean_iou.item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status
    
    def compute_losses_MLP(self, pred_dict, gt_dict):
        K=30
        reg_max=16
        # Get GT boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)

        # Get pred
        pred_boxes = pred_dict['pred_boxes']
        pred_dist = pred_dict['pred_dist'].reshape(-1, 4*(reg_max+1))
        pred_query_ind = pred_dict['query_ind']
        query_map = torch.zeros([self.bs, self.search_feat_size * self.search_feat_size], device=pred_query_ind.device)
        query_map.scatter_(1, pred_query_ind, 1)
        query_map_2d = query_map.reshape(self.bs, self.search_feat_size, self.search_feat_size)
        gt_box_map_2d =  generate_cls_map(box_xywh_to_xyxy(gt_bbox), self.search_feat_size)
        query_inbox_map_2d = gt_box_map_2d * query_map_2d

                                           
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        # Grep positive & negative predicts
        pos_box_logits = torch.gather(query_inbox_map_2d.flatten(1), 1, pred_query_ind)
        pos_box_ind = pos_box_logits.reshape(-1).nonzero().reshape(-1)
        neg_box_ind = (pos_box_logits.reshape(-1) == 0).nonzero().reshape(-1)
        pred_box_pos = pred_boxes[pos_box_ind]
        pred_dist_pos = pred_dist[pos_box_ind]
        pred_dist_neg = pred_dist[neg_box_ind]
        
        # Making positve ltrb targets
        inside_ind = torch.nonzero(query_inbox_map_2d.flatten(1))
        gt_box_pos = gt_bbox[inside_ind[:,0],:]
        gt_box_pos_vec = box_xywh_to_xyxy(gt_box_pos).clamp(min=0.0,max=1.0)
        target_ltrb_pos =  bbox2distance(grid_center_2d(query_inbox_map_2d, self.search_feat_size),
            gt_box_pos_vec * self.search_feat_size, max_dis=self.search_feat_size).reshape(-1)# dont norm to 1
        # target_dist_neg = pred_dist_neg.new_zero(pred_dist_neg.shape)
        
        # Box loss
        try:
            giou_loss, _ = self.objective['giou'](pred_box_pos, gt_box_pos_vec)
            giou_loss = giou_loss.mean()
        except:
            giou_loss = torch.as_tensor(0.0).to("cuda")
            _ = torch.as_tensor(0.0).to("cuda")
        l1_loss = self.objective['l1'](pred_box_pos, pred_box_pos)  # (BN,4) (BN,4)

        # Distribution loss
        df_loss = self.objective['dfl'](pred_dist_pos.reshape(-1,17), target_ltrb_pos, weight=None)
        # neg_loss = self.
                
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss\
            + self.loss_weight['l1'] * l1_loss\
            + df_loss

            # + ctr_loss * 0 \
        
        # status for log
        mean_iou = _.detach().mean()
        # mean_giou_loss = top_giou.detach().mean()
        
        status = {
                    "IoU": mean_iou.item(),
                #   "IoU_T": mean_iou.item(),
                #   "Coord":pred_boxes_vec[0].detach().cpu()
                    }
        return loss, status



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
