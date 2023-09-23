from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-7)
        _prediction = torch.clamp(1-prediction, 1e-7)
        positive_loss = torch.log(prediction) * torch.pow(_prediction, self.alpha) * positive_index
        negative_loss = torch.log(_prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss

# For corner head
class DistributionFocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(DistributionFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.gt(0).float()
        negative_index = target.eq(0).float()
        

        # # negative_weights = 1 for all neg
        # negative_weights = torch.pow(1 - target, self.beta)
        # # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, torch.finfo(prediction.dtype).smallest_normal)
        loss = torch.log(prediction) * target * positive_index
        loss = -loss.sum()
        target_ = torch.clamp(target, torch.finfo(target.dtype).smallest_normal)
        min_loss = torch.log(target_) * target * positive_index
        # _prediction = torch.clamp(1-prediction, 1e-7)
        # positive_loss = torch.log(prediction) * torch.pow(_prediction, self.alpha) * positive_index
        # negative_loss = torch.log(_prediction) * torch.pow(prediction,
        #                                                       self.alpha) * negative_weights * negative_index

        # num_positive = positive_index.float().sum()
        # positive_loss = positive_loss.sum()
        # negative_loss = negative_loss.sum()

        # if num_positive == 0:
        #     loss = -negative_loss
        # else:
        #     loss = -(positive_loss + negative_loss) / num_positive

        return loss + min_loss.sum()    

class DFL(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(DFL, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, 
            pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None):
        disl = label.long()
        disr = disl + 1

        wl = disr.float() - label
        wr = label - disl.float()
        # if disl.max()>=16 or disr.max()>=16:
        #     print(' ')
        loss = F.cross_entropy(pred, disl, reduction='none') * wl \
            + F.cross_entropy(pred, disr, reduction='none') * wr
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

class QFL(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(QFL, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self,
          pred,          # (n, 1)
          label,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          score,         # (n) reg target 0-1, only positive is good
          weight=None,
          beta=2.0,
          reduction='mean',
          avg_factor=None):
        # all goes to 0
        pred_sigmoid = pred.sigmoid()
        zerolabel = pred_sigmoid.new_zeros(pred.shape)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction='none') * pred_sigmoid.pow(beta)

        # label = label - 1
        pos = (label > 0).nonzero().squeeze(1)
        
        # positive goes to bbox quality
        pt = score[pos] - pred_sigmoid[pos, 0]
        loss[pos,0] = F.binary_cross_entropy_with_logits(
            pred[pos,0], score[pos], reduction='none') * pt.pow(beta)

        # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

class FL(nn.Module, ABC):
    def __init__(self, alpha=0.25, beta=2):
        super(FL, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self,
          pred,          # (n, 1)
          label,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          score=None,
          weight=None,
          beta=2.0,
          reduction='mean',
          avg_factor=None):
        # all goes to 0
        pred_sigmoid = pred.sigmoid()
        # zerolabel = pred_sigmoid.new_zeros(pred.shape)
        ce_loss = F.binary_cross_entropy_with_logits(pred, label, weight=score, reduction='none')
        p_t = pred_sigmoid * label + (1 - pred_sigmoid) * (1 - label)
        loss = ce_loss * ((1 - p_t) ** self.beta)
        if self.alpha >= 0:
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)
            loss = alpha_t * loss
        return loss


class old_DistributionFocalLoss(nn.Module, ABC):
    def __init__(self):
        super(DistributionFocalLoss, self).__init__()
    def forward(self, pred, label, size):
        size = int(size)
        bs, n_feat, _ = pred.shape
        pred_map = pred.view(bs, n_feat, size+1, size+1)
        x_distri = pred_map.sum(-2)
        y_distri = pred_map.sum(-1)
        label_grid = size * label
        dis_left = label_grid.long().clamp(max=size-1)
        dis_right = dis_left + 1
        weight_left = dis_right.float() - label_grid
        weight_right = label_grid - dis_left.float()
        # bottom and ceiling
        indexs_X = torch.stack((dis_left.view(bs,2,2)[:, :, 0], dis_right.view(bs,2,2)[:, :, 0]), -1)
        indexs_Y = torch.stack((dis_left.view(bs,2,2)[:, :, 1], dis_right.view(bs,2,2)[:, :, 1]), -1)

        pred_X = x_distri.gather(-1, indexs_X)
        pred_Y = y_distri.gather(-1, indexs_Y)
        target_X = torch.stack((weight_left.view(bs,2,2)[:, :, 0], weight_right.view(bs,2,2)[:, :, 0]), -1)
        target_Y = torch.stack((weight_left.view(bs,2,2)[:, :, 1], weight_right.view(bs,2,2)[:, :, 1]), -1)
        loss = F.cross_entropy(pred_X[:,0], target_X[:,0], reduction='none')  \
            + F.cross_entropy(pred_Y[:,1], target_Y[:,1], reduction='none') \
            + F.cross_entropy(pred_Y[:,0], target_Y[:,0], reduction='none') \
            + F.cross_entropy(pred_Y[:,1], target_Y[:,1], reduction='none') \
            
        return loss.mean()

class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss