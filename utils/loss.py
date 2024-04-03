import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from torch.nn.modules.loss import _Loss

# CORE: COnsistent REpresentation Learning for Face Forgery Detection, CVPRW 22
# Revise by Sun
class ConsistencyCos(nn.Module):
    def __init__(self, device='cpu'):
        super(ConsistencyCos, self).__init__()
        self.mse_fn = nn.MSELoss()
        self.device = device

    def forward(self, feat):
        feat = nn.functional.normalize(feat, dim=1)
        feat_0 = feat[:int(feat.size(0)/2),:]
        feat_1 = feat[int(feat.size(0)/2):,:]
        cos = torch.einsum('nc,nc->n', [feat_0, feat_1]).unsqueeze(-1)
        labels = torch.ones((cos.shape[0],1), dtype=torch.float, requires_grad=False)
        loss = self.mse_fn(cos, labels.to(self.device))
        return loss

class PolyBCELoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.bce_loss = self.bce(input, target)
        pt = torch.sigmoid(input) 
        pt = torch.where(target==1,pt,1-pt)
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)
    


#
class Arcface_Head(nn.Module):
    def __init__(self, embedding_size=1792, num_classes=2, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output  *= self.s
        return output


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='sum'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class BCE_FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, logits=True, reduce=True):
        super(BCE_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, predict, label):
        cls_score = predict
        label = label
        if self.logits:  # TODO
            BCE_loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduce=False
            )
        else:
            BCE_loss = F.binary_cross_entropy(cls_score, label, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features, device):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight.to(x.device), dim=0))
        return cos_theta.clamp(-1, 1)


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.5, s=30, t=1.):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m #cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        # print(target.data.view(-1, 1))
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            return F.cross_entropy(self.s*output, target), self.s*output

        if self.t > 1:
            h_theta = self.t - 1 + self.t*cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s*output, target)

        return focal_loss(F.cross_entropy(self.s*output, target, reduction='none'), self.gamma), self.s*output





class BinaryFocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weights = weights      # [class_0, class_1] và class_0 + class_1 = 1
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, target):    # pred: (batch_size, ), target: (batch_size, )
        pred = pred.clamp(self.eps, 1 - self.eps)
        if self.weights is not None:
            loss = self.weights[1] * ((1 - pred) ** self.gamma) * target * torch.log(pred) + self.weights[0] * (pred ** self.gamma) * (1 - target) * torch.log(1 - pred)
        else:
            loss = ((1 - pred) * self.gamma) * target * torch.log(pred) + (pred ** self.gamma) * (1 - target) * torch.log(1 - pred)
        return torch.mean(torch.neg(loss))

class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.
    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.
        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.
        Returns:
            torch.Tensor: The calculated loss.
        """
        ret = self._forward(*args, **kwargs)
        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            ret *= self.loss_weight
        return ret


class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.
    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT'])
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def _forward(self, outputs, samples, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The class score.
            samples (dict): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if cls_score.size() == label.size():
            # calculate loss for soft labels

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, (
                'For now, no extra args are supported for soft label, '
                f'but get {kwargs}'
            )

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                lsm = lsm * self.class_weight.unsqueeze(0).to(cls_score.device)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0).to(cls_score.device) * label
                )
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert (
                    'weight' not in kwargs
                ), "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT'])
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def _forward(self, outputs, samples, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The class score.
            samples (dict): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.
        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if self.class_weight is not None:
            assert (
                'weight' not in kwargs
            ), "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_score, label, **kwargs
        )
        return loss_cls


class MSELoss(BaseWeightedLoss):
    """MSE Loss
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.

    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.mse = nn.MSELoss()

    def _forward(self, pred_mask, gt_mask, **kwargs):  # TODO samples
        loss = self.mse(pred_mask, gt_mask)
        return loss

import numpy as np


class WeaklyMaskLoss(nn.Module):
    """MSE Loss"""

    def __init__(self, device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.device = device
        
    def forward(self, pred_mask, labels, img_size=256, **kwargs):  # TODO samples
        B, C, H, W = pred_mask.shape
        idx = torch.where(labels==1)
        cal_pred = pred_mask[idx]
        nums = cal_pred.size(0)
        gt_mask = torch.zeros((nums,1, img_size, img_size)).to(self.device)
        loss = self.mse(cal_pred, gt_mask)
        return loss


class ICCLoss(BaseWeightedLoss):
    """Contrastive Loss
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self):
        super().__init__(loss_weight=1)

    def _forward(self, feature, label, **kwargs):  # TODO samples
        # size of feature is (b, 1024)
        # size of label is (b)
        C = feature.size(1)
        label = label.unsqueeze(1)
        label = label.repeat(1, C)
        # print(label.device)
        label = label.type(torch.BoolTensor).cuda()

        res_label = torch.zeros(label.size(), dtype=label.dtype)
        res_label = torch.where(label == 1, 0, 1)
        res_label = res_label.type(torch.BoolTensor).cuda()

        # print(label, res_label)
        pos_feature = torch.masked_select(feature, label)
        neg_feature = torch.masked_select(feature, res_label)

        # print('pos_fea: ', pos_feature.device)
        # print('nge_fea: ', neg_feature.device)
        pos_feature = pos_feature.view(-1, C)
        neg_feature = neg_feature.view(-1, C)

        pos_center = torch.mean(pos_feature, dim=0, keepdim=True)

        # dis_pos = torch.sum((pos_feature - pos_center)**2) / torch.norm(pos_feature, p=1)
        # dis_neg = torch.sum((neg_feature - pos_center)**2) / torch.norm(neg_feature, p=1)
        num_p = pos_feature.size(0)
        num_n = neg_feature.size(0)
        pos_center1 = pos_center.repeat(num_p, 1)
        pos_center2 = pos_center.repeat(num_n, 1)
        dis_pos = F.cosine_similarity(pos_feature, pos_center1, eps=1e-6)
        dis_pos = torch.mean(dis_pos, dim=0)
        dis_neg = F.cosine_similarity(neg_feature, pos_center2, eps=1e-6)
        dis_neg = torch.mean(dis_neg, dim=0)

        loss = dis_pos - dis_neg

        return loss


class FocalLoss(BaseWeightedLoss):
    def __init__(self, weight, alpha=1, gamma=2, isLogits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = isLogits
        self.reduce = reduce

    def _forward(self, cls_score, label, **kwargs):
        if self.logits:  # TODO
            BCE_loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduce=False
            )
        else:
            BCE_loss = F.binary_cross_entropy(cls_score, label, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)



class Multi_attentional_Deepfake_Detection_loss(nn.Module):
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def forward(self, loss_pack, label):
        if 'loss' in loss_pack:
            return loss_pack['loss']
        loss = (
            self.loss_cfg['ENSEMBLE_LOSS_WEIGHT'] * loss_pack['ensemble_loss']
            + self.loss_cfg['AUX_LOSS_WEIGHT'] * loss_pack['aux_loss']
        )
        if self.loss_cfg['AGDA_LOSS_WEIGHT'] != 0:
            loss += (
                self.loss_cfg['AGDA_LOSS_WEIGHT']
                * loss_pack['AGDA_ensemble_loss']
                + self.loss_cfg['MATCH_LOSS_WEIGHT'] * loss_pack['match_loss']
            )
        return loss

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

    
class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes, 
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD] 
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)



        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 epsilon: float = 1.0,
                 alpha: float = 1,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

import torch
import torch.nn as nn


class SingleCenterLoss(nn.Module):
    """
    Single Center Loss
    
    Reference:
    J Li, Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection, CVPR 2021.
    
    Parameters:
        m (float): margin parameter. 
        D (int): feature dimension.
        C (vector): learnable center.
    """
    def __init__(self, m = 0.3, D = 1000, device='cpu'):
        super(SingleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.l2loss = nn.MSELoss(reduction = 'none')
        # if self.use_gpu:
        #     self.C = nn.Parameter(torch.randn(self.D).cuda())
        # else:
        self.C = nn.Parameter(torch.randn(self.D).to(device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        eud_mat = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))

        labels = labels.unsqueeze(1)

        real_count = labels.sum()
        
        dist_real = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()
        dist_fake = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()

        if real_count != 0:
            dist_real /= real_count

        if real_count != batch_size:
            dist_fake /= (batch_size - real_count)

        max_margin = dist_real - dist_fake + self.margin

        if max_margin < 0:
            max_margin = 0

        loss = dist_real + max_margin

        return loss

if __name__ == "__main__":
    dummy = torch.rand((48,1, 64, 64))
    label = torch.cat((torch.ones(24), torch.zeros(24)), dim=0)
    loss = WeaklyMaskLoss()
    _ = loss(dummy, label)


