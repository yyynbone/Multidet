import torch
import torch.nn as nn

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true, cls_weight=None):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits   128,3,640,640
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)     # 128,3,640,640
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)      # 128,3,640,640
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        #modefy
        if (cls_weight != None) and cls_weight.dim() == loss.dim():
            # loss *= torch.tensor(self.cls_weight,device=loss.device).float()
            loss *= cls_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = nn.functional._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def cross_entropy(pred,
                  label,
                  reduction='mean',
                  class_weight=None,
                  ignore_index=-100,
                  **kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = nn.functional.cross_entropy(
        pred,
        label.flatten(),
        weight=class_weight/ sum(class_weight),
        reduction='none',
        ignore_index=ignore_index)
    loss = reduce_loss(loss, reduction)
    return loss

def binary_cross_entropy(pred,
                  label,
                  reduction='mean',
                  class_weight=None,
                  pos_weight=None,
                  loss_style=0,
                  **kwargs):
    """
    difference in loss_style, the result is loss.mean():
        none:  pos 1, 10 ,100 (0.74, 2.2, 164.8) to (0.70, 2.88, 243.7)
               if this, when pos_weight gain, although the same pred, but loss decrease, so we modifed to neg_pos_weight
        neg_pos_weight: pos 1, 10 ,100 (0.74, 0.40, 0.33) to (0.69, 0.52,0.49)
        pos_sample_aug: pos 1, 10 ,100 (0.74, 0.55, 0.49) to (0.7, 0.52, 0.487)
                        why? because if label 0, pred 0.1, after sigmoid, pred is 0.525, so loss is 0.74;
                        but if label 1., pred 0.5, after sigmoid, pred is 0.6225, , so loss is 0.474
                        changed  sharply, if pos label increase, the divident become bigger
    :param pred:
    :param label:
    :param reduction:  mean, None, sum
    :param class_weight: multi_class weight for labels, eg: all category is 4 (0 to 3),class_weight.shape=4,
        if class_weight=[0.1, 0.1, 0.6,0.2], one label of image if [1, 1, 0, 1], pred=[0.9, 0.2, 0.1,0.8], loss = class_weight * norm(pred-label)
    :param pos_weight:
    :param loss_style: loss calculate style, [0, 1, 2] denote [others,'neg_pos_weight', 'pos_sample_aug']
    :param kwargs:
    :return:
    """
    if pos_weight is  None:
        pos_weight = torch.tensor([1.], device=pred.device)
    loss = nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
            class_weight,
            pos_weight=pos_weight,
            reduction='none')

    if loss_style == 1:
        loss = 2 * loss / (1 + pos_weight)
    elif loss_style == 2:
        loss = loss * label.shape[0] / (label.shape[0] + (pos_weight - 1) * label.sum())
    loss = reduce_loss(loss, reduction)
    return loss
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 # use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        # assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        # self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        # elif self.use_mask:
        #     self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                ignore_index=None,
                **kwargs):
        if self.class_weight is not None:

            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.cls_criterion(
            cls_score,
            label,
            class_weight=class_weight,
            reduction=self.reduction,
            ignore_index=ignore_index,
            **kwargs)
        # print("now loss cls is ", loss_cls)
        loss_cls *= self.loss_weight
        # print("after lose weight now loss cls is ", loss_cls)
        return loss_cls
# modify  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

