import torch
import numpy as np
import math
import json
import os
from pathlib import Path
from utils.label_process import xyxy2xywh, xywh2xyxy
import time
import torchvision
from utils.plots import plot_pr_curve, plot_confusionmatrics, print_log
from utils.cal_coco import select_score_from_json, visual_return
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def box_iou(box1, box2, eps=1E-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter+eps)  # iou = inter / (area1 + area2 - inter)

def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def box_iof(box1, box2):
    """
    Return intersection-over-min area (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # Intersection over the smaller area
    return inter / torch.min(area1[:, None], area2[None])

# def div_area_idx(area, div_area, area_idx):
#     """
#     divide the area by pix_node and return area_idx
#     :param area: tensor or numpy
#     :param div_area:
#     :param area_idx: tensor bool or numpy bool
#     :return:
#     """
#     if len(div_area) == area_idx.shape[1]-2:
#         if area_idx.dim()==3:
#             area = area[:, None].repeat(1, area_idx.shape[2])
#         area_idx[:, 1] = area <= div_area[0] ** 2
#         for div_node in range(len(div_area) - 1):
#             area_idx[:, div_node + 2] = (
#                 np.logical_and(area > div_area[div_node] ** 2, area <= div_area[div_node + 1] ** 2).bool())
#         area_idx[:, -1] = area > div_area[-1] ** 2
#
#     return area_idx
#
# def process_batch(detections, labels, iouv, div_area=None):
#     """
#     Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
#     Arguments:
#         detections (Array[N, 6]), x1, y1, x2, y2, conf, class
#         labels (Array[M, 5]), class, x1, y1, x2, y2
#     Returns:
#         correct (Array[N, 10]), for 10 IoU levels
#     """
#     div_class_num = 1
#     if isinstance(div_area, (list, tuple)):
#         assert len(div_area)>0, "now the divide node of area  are recommend to be 2,so we divide to small media and large"
#         div_class_num+=len(div_area)+1
#
#     correct = torch.zeros(detections.shape[0], div_class_num, iouv.shape[0], dtype=torch.bool, device=iouv.device)
#     area_idx = torch.ones(labels.shape[0], div_class_num, dtype=torch.bool)
#
#     iou = box_iou(labels[:, 1:], detections[:, :4])
#     x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
#     if x[0].shape[0]:
#         matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
#         if x[0].shape[0] > 1:
#             matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#             # matches = matches[matches[:, 2].argsort()[::-1]]
#             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#
#         matches = torch.Tensor(matches).to(iouv.device)
#         correct[matches[:, 1].long()] = (matches[:, 2:3] >= iouv)[:, None, :].repeat(1, div_class_num, 1)
#         if isinstance(div_area, (list, tuple)):
#             # first we assigned all pred box to area level, so that the pred that not matched can also assigned.
#             pred_box = xyxy2xywh(detections[:, :4])
#             pred_area = (pred_box[:, 2] * pred_box[:, 3]).cpu()
#             correct = div_area_idx(pred_area, div_area, correct)
#             tbox = xyxy2xywh(labels[:, 1:])
#             area = (tbox[:, 2] * tbox[:, 3]).cpu()
#             area_idx = div_area_idx(area, div_area, area_idx)
#             for area_id_idx in range(1, div_class_num):
#                 area_id_bool = area_idx[:, area_id_idx]
#                 area_ids = torch.arange(len(area_id_bool))[area_id_bool]
#                 #np and tensor broadcast is not the same
#                 area_bool = np.logical_or.reduce(matches[:, :1].cpu()==area_ids, axis=1)  # should be (match_num, 1)
#                 area_bool = area_bool[...,None].to(iouv.device).repeat(1, 10)
#                 # correct[matches[:, 1].long(),area_id_idx+1] = torch.logical_and(correct[matches[:, 1].long(),area_id_idx+1], area_bool )
#                 correct[matches[:, 1].long(), area_id_idx] = area_bool.bool()
#
#     return correct, area_idx

# def area_filter(array, e_class=4, filter_idx=1):
#     assert array.shape[0]%e_class==0, 'must be divided by e_class'
#     idx = []
#     for i in range(array.shape[0]):
#         if i%e_class !=filter_idx :
#             idx.append(i)
#     return array[idx]

def area_filter(array, filter_from=2):
    return array[:, filter_from:]

def div_area_idx(area, div_area, area_idx):
    """
    divide the area by pix_node and return area_idx, and only divide the grid which is not True
    :param area: tensor or numpy
    :param div_area:
    :param area_idx: tensor bool or numpy bool
    :return:
    """
    device = None
    if len(div_area) == area_idx.shape[1]-2:
        if isinstance(area, torch.Tensor):
            area = area.cpu()
        if isinstance(area_idx, torch.Tensor):
            device = area_idx.device
            area_idx = area_idx.bool().cpu()
        if area_idx.dim()==3:
            #dim_flag = area_idx.shape[-1]
            # process the pred correct
            area = area[:, None].repeat(1, area_idx.shape[-1])
            # area = area[:, None].repeat(1, area_idx.shape[2])

        # 将已配对的排除掉，故只要要处理，为False的
        idx = (~area_idx[:, 0])
        area_idx[:, 1][idx] = (area <= div_area[0] ** 2)[idx]
        for div_node in range(len(div_area) - 1):

            area_idx[:, div_node + 2][idx] = (np.logical_and(area > div_area[div_node] ** 2,
                                            area <= div_area[div_node + 1] ** 2).bool())[idx]

        area_idx[:, -1][idx] = (area > div_area[-1] ** 2)[idx]
    check_idx(area_idx)
    return area_idx.to(device) if device is not None else area_idx

def check_idx(area_idx):
    if area_idx.dim() == 3:
        area_idx = area_idx[:, :, 0]
    assert area_idx.shape[0] == sum([area_idx[:, i].sum() for i in range(1, area_idx.shape[1])])


def process_batch(detections, labels, iouv, div_area=None):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    div_class_num = 1
    if isinstance(div_area, (list, tuple)):
        assert len(div_area)>0, "now the divide node of area  are recommend to be 2,so we divide to small media and large"
        div_class_num+=len(div_area)+1

    correct = torch.zeros(detections.shape[0], div_class_num, iouv.shape[0], dtype=torch.bool, device=iouv.device)
    area_idx = torch.zeros(labels.shape[0], div_class_num, dtype=torch.bool)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    matches = torch.zeros(0, 3)  # [label, detection, iou]

    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long(), 0] = (matches[:, 2:3] >= iouv)
    # if matches.shape[0]!=labels.shape[0]:
    #     print(matches, labels, 'loss matched')
    if isinstance(div_area, (list, tuple)):
        tbox = xyxy2xywh(labels[:, 1:])
        area = (tbox[:, 2] * tbox[:, 3]).cpu()
        area_idx = div_area_idx(area, div_area, area_idx)
        for area_id_idx in range(1, div_class_num):
            area_id_bool = area_idx[:, area_id_idx]
            area_ids = torch.arange(len(area_id_bool))[area_id_bool]
            #np and tensor broadcast is not the same
            area_bool = np.logical_or.reduce(matches[:, :1].cpu()==area_ids, axis=1)  # should be (match_num, 1)
            area_bool = area_bool[...,None].to(iouv.device).repeat(1, iouv.shape[0])
            # correct[matches[:, 1].long(),area_id_idx+1] = torch.logical_and(correct[matches[:, 1].long(),area_id_idx+1], area_bool )
            correct[matches[:, 1].long(), area_id_idx] = area_bool.bool()
        # because we calculate the number, first we should match the gt,  then wo assigned the pred which is not matched.
        # last  we assigned all pred box to area level, so that the pred that not matched can also assigned.
        pred_box = xyxy2xywh(detections[:, :4])
        pred_area = (pred_box[:, 2] * pred_box[:, 3])
        correct = div_area_idx(pred_area, div_area, correct)
    return correct, area_idx, matches

def fitness(x):
    # Model fitness as a weighted combination of metrics
    metric_num = x.shape[1]
    if metric_num>6:
        w = [0.1, 0.3, 0.1, 0.9, 0.1, 0.1, 0.3]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (x[:, :7] * w).sum(1)
    else:
        w = [0.1, 0.3, 0.1, 0.9]
        return (x[:, :4] * w).sum(1)

def calcu_per_class(tp: object, conf: object, pred_cls: object, target_cls: object, area_idx: object = None, plot: object = False, save_dir: object = '.', names: object = (),
                    conf_ts: object = 0.1,
                    eps: object = 1e-16) -> object:
    """ Compute the average precision, recall and precision

    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    if not isinstance(conf_ts, (list, tuple)):
        conf_ts = [conf_ts]
    conf_length = len(conf_ts)
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # Find unique classes
    # unique_classes, nt = np.unique(target_cls, return_counts=True)
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    div_class_num = tp.shape[1]
    ap = np.zeros((nc, div_class_num, tp.shape[2]))
    lev_tn = np.zeros((nc,  div_class_num))
    p, r = np.zeros((conf_length, nc,  div_class_num)), np.zeros((conf_length, nc, div_class_num))
    pred_truth, lev_pn = np.zeros((conf_length, nc,  div_class_num)), np.zeros((conf_length, nc, div_class_num))
    area_idx[:, 0] = True
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        for idx in range(div_class_num):
            tp_perc_area = tp[i, 0]
            conf_cls = conf[i]
            if idx:
                area_matched_idx = tp[i, idx, 0]
                tp_perc_area = tp_perc_area[area_matched_idx]
                conf_cls = conf_cls[area_matched_idx]

            n_l = (target_cls[area_idx[:, idx]]==c).sum()
            lev_tn[ci, idx] = n_l
            n_p = len(tp_perc_area[:, 0])  # number of predictions
            for li, conf_t in enumerate(conf_ts):
                pred_truth[li, ci, idx] = tp_perc_area[conf_cls >= conf_t][:, 0].sum() if n_l else 0.
                # -1 means the max ,0 means iou=0.5, precision shape [n,10]
                lev_pn[li, ci, idx] = tp_perc_area[conf_cls >= conf_t].shape[0] if n_p else 0.
            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp_perc_area).cumsum(0)
                tpc = tp_perc_area.cumsum(0)
                # Recall
                recall = tpc / (n_l + eps)  # recall curve
                # Precision
                precision = tpc / (tpc + fpc)  # precision curve

                # AP from recall-precision curve
                for j in range(tp.shape[2]):
                    ap[ci, idx, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0 and idx == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5


                for li, conf_t in enumerate(conf_ts):
                    precision_c = precision[conf_cls >= conf_t]
                    recall_c = recall[conf_cls >= conf_t]
                    p[li, ci, idx] = precision_c[-1, 0]  if precision_c.shape[0] else 0.  # -1 means the max ,0 means iou=0.5, precision shape [n,10]
                    r[li, ci, idx] = recall_c[-1, 0] if recall_c.shape[0] else 0.

    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot and len(py):
        plot_pr_curve(px, py, ap[:,0,:], Path(save_dir) / 'PR_curve.png', names)

    return p, r, pred_truth, lev_pn, ap, unique_classes.astype('int32'), lev_tn

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        plot_confusionmatrics(self.matrix, self.nc, normalize, save_dir, names)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300,max_nms=30000, merge = False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates  #b, k, 1

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    # max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def nms_iof(prediction, iof_thres=0.8):
    """
    filter the box with iof and score
    :param prediction:
    :param iof_thres:
    :return:
    """
    output = []
    for xi, nms_out in enumerate(prediction):
        # filter box with iof, then select box via score
        nms_box = nms_out[:, :4]
        iof = box_iof(nms_box, nms_box) > iof_thres  # iou matrix , i*i
        nms_score = nms_out[:, 4].cpu()
        cls = nms_out[:, -1]
        cls_arr = cls[:,None] == cls[None]
        iof_cls =  np.logical_and(cls_arr.cpu(), iof.cpu())  # numpy tensor must in cpu
        m_i, n_i = torch.triu(iof_cls.long() - torch.eye(nms_box.shape[0])).nonzero(as_tuple=False).T
        filter_id = set(torch.where(nms_score[m_i] < nms_score[n_i], m_i, n_i).tolist())
        maintain_id = [ids for ids in range(len(nms_out)) if ids not in filter_id]
        output.append(nms_out[maintain_id])
    return output

def non_max_suppression_with_iof(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300,max_nms=30000, merge = False, iof_nms=False):
    prediction = non_max_suppression(prediction, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes,
                    agnostic=agnostic, multi_label=multi_label, labels=labels, max_det=max_det,
                    max_nms=max_nms, merge=merge)

    return nms_iof(prediction, iof_thres=0.8) if iof_nms else prediction

def classify_match(pred_out, class_label, conf_ts=0.5, logger=None):
    if pred_out.shape[1] > 1:
        pred_out = pred_out[..., -1:]
    if not isinstance(conf_ts, (list, tuple)):
        conf_ts = [conf_ts]
    ps, rs, accus = [], [], []
    for conf_t in conf_ts:
        out = np.where(pred_out > conf_t, 1, 0)
        matched = out[:, 0] == class_label[:, 0]
        match_num = matched.sum()
        all_num = len(class_label)
        accu = match_num /max(all_num, 0.01)
        accus.append(accu)
        s = ('%20s' + '%11s' * 5) % ('Class', 'Labels', 'P_num', 'R_num', 'P', 'R')
        print_log(s, logger)
        pf = '%20s' + '%11i' * 3 + '%11.3g' * 2  # print format
        print_log(pf % (f'all in conf {conf_t}', all_num, all_num, match_num, accu, accu), logger)
        for i,cls in enumerate(['background', 'object']):
            obj_num = (class_label[:, 0] == i).sum()
            pred_num = (out[:, 0] == i).sum()
            match_num = (out[matched] == i).sum()
            p = match_num/max(pred_num,0.01)
            r = match_num/max(obj_num,0.01)
            ps.append(p)
            rs.append(r)
            print_log(pf % (cls, obj_num, pred_num, match_num, p, r), logger)
        # logger_func(f"there are bg number {bg_num},  object number {obj_num}, pred object is {pred_num}, obj_matched {obj_match_num},  so P/{p:.3f} R/{r:.3f}")
    return ps, rs, accus

def det_coco_calculate(weights, data, task, save_dir, logger, jdict, last_conf, visual_matched):
    is_coco = isinstance(data.get('coco'), str) # and data['coco'].endswith('.json')  # COCO dataset
    w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    # anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
    # anno_json = str(Path(data.get('path', '../coco')) / f'COCO/annotation/{task}.json')  # annotations json

    pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json

    convert_small2big_json(jdict)
    ######### for Imageid  of filename 2 Imageid ################
    if is_coco:
        try:
            anno_json = str(Path(data['coco']))
            with open(anno_json, 'r') as f:
                images = json.load(f)['images']
            image_file_id = {}
            for img_info in images:
                file_name_no_suffix = img_info['file_name'].split('.')[0]
                image_file_id[file_name_no_suffix] = [img_info['file_name'], img_info['id']]
            for pic_info in jdict:
                pic_info['image_name'], pic_info['image_id'] = image_file_id[pic_info['image_name']]
        except Exception as e:
            print_log(f'open anno file ERROR: {e}', logger)

    with open(pred_json, 'w') as f:
        json.dump(jdict, f, indent=4)
    if is_coco:
        print_log(f'\nEvaluating pycocotools mAP... saving {pred_json}...', logger)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api

            pred_json_c = select_score_from_json(pred_json, score_thresh=0.1)

            pred = anno.loadRes(pred_json_c)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.params.maxDets = [10, 300, 5000]
            eval.params.iouThrs = np.linspace(.05, 0.95, int(np.round((0.95 - .05) / .1)) + 1, endpoint=True)
            # if is_coco:
            #     eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()

            def eval_summarize(eval, ap=1, iouThr=None, areaRng='all', maxDets=100):
                p = eval.params
                iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
                typeStr = '(AP)' if ap == 1 else '(AR)'
                iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)

                aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
                if ap == 1:
                    # dimension of precision: [TxRxKxAxM]
                    s = eval.eval['precision']
                    # IoU
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, :, aind, mind]
                else:
                    # dimension of recall: [TxKxAxM]
                    s = eval.eval['recall']
                    if iouThr is not None:
                        t = np.where(iouThr == p.iouThrs)[0]
                        s = s[t]
                    s = s[:, :, aind, mind]
                if len(s[s > -1]) == 0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s > -1])
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s
            eval_summarize(eval, 1, iouThr=.2, maxDets=eval.params.maxDets[2])
            eval_summarize(eval, 0, iouThr=.2, maxDets=eval.params.maxDets[2])
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

            # with open(data[task], 'r') as f:
            #     img0 = f.readline()
            # img_prefix = os.path.split(img0)[0]
            img_prefix = str(data[task])

            for conf_t in last_conf:
                print_log(
                    f"##########################\nnow we collect and visual result with iou thresh of "
                    f"{eval.params.iouThrs[0]} and conf thresh of {conf_t} in origin", logger=logger)

                visual_return(eval, anno, save_dir, img_prefix, class_area=None, score_thresh=conf_t,
                              logger=logger, save_visual=visual_matched)
                visual_matched = False

        except Exception as e:
            print_log(f'pycocotools unable to run: {e}', logger)
        
def det_calculate(stats, div_area, nc, last_conf, save_dir, names, plots,verbose, logger):
    s = ('%20s' + '%11s' * 7) % ('Class', 'Labels', 'R_num', 'P_num', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, mp, mr, map50, map =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ps, rs, pred_truths, lev_nps, ap, ap_class, nt = calcu_per_class(*stats, plot=plots, save_dir=save_dir, names=names,
                                                                     conf_ts=last_conf)
    ap50, ap = ap[..., 0], ap.mean(2)  # AP@0.5, AP@0. 5:0.95
    for li, conf_t in enumerate(last_conf):
        p, r, pred_truth, lev_np = ps[li], rs[li], pred_truths[li], lev_nps[li]
        # Print results
        print_log(s, logger)
        pf = '%20s' + '%11i' * 3 + '%11.3g' * 4  # print format
        all_tp, all_p, all_t = pred_truth[:, 0].sum(), lev_np[:, 0].sum(), nt[:, 0].sum()

        mp, mr = all_tp / max(all_p, 1e-16), all_tp / max(all_t, 1e-16)
        map50, map = ap50[:, 0].mean(), ap[:, 0].mean()
        print_log(pf % (f'all_conf_{conf_t}', all_t, all_tp, all_p, mp, mr, map50, map), logger)
        if isinstance(div_area, (list, tuple)):
            for div_i, div_ang in enumerate(div_area):
                f_t, f_tp, f_p = nt[..., div_i+1], pred_truth[..., div_i+1], lev_np[..., div_i+1]
                f_map50, f_map = ap50[..., div_i+1], ap[..., div_i+1]
                f_map50 = (f_map50 * f_p).sum() / max(f_p.sum(), 1e-16)
                f_map = (f_map * f_p).sum() / max(f_p.sum(), 1e-16)
                f_t, f_tp, f_p = f_t.sum(), f_tp.sum(), f_p.sum()
                f_mp, f_mr = f_tp / max(f_p, 1e-16), f_tp / max(f_t, 1e-16)
                print_log(pf % (f'all_under_{div_area[div_i]}', f_t, f_tp, f_p, f_mp, f_mr, f_map50, f_map), logger)

            f_t, f_tp, f_p = area_filter(nt, -1), area_filter(pred_truth, -1), area_filter(lev_np, -1)
            f_map50, f_map = area_filter(ap50, -1), area_filter(ap, -1)
            f_map50 = (f_map50 * f_p).sum() / max(f_p.sum(), 1e-16)
            f_map = (f_map * f_p).sum() / max(f_p.sum(), 1e-16)
            f_t, f_tp, f_p = f_t.sum(), f_tp.sum(), f_p.sum()
            f_mp, f_mr = f_tp / max(f_p, 1e-16), f_tp / max(f_t, 1e-16)
            print_log(pf % (f'all_over_{div_area[-1]}', f_t, f_tp, f_p, f_mp, f_mr, f_map50, f_map), logger)
        print_log("-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -", logger)
        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                if c in names:
                    print_log(pf % (
                    names[c], nt[i, 0], pred_truth[i, 0], lev_np[i, 0], p[i, 0], r[i, 0], ap50[i, 0], ap[i, 0]), logger)
                    if isinstance(div_area, (list, tuple)):
                        for div_i, div_ang in enumerate(div_area):
                            print_log(pf % (f'{names[c]}_under_{div_ang}', nt[i, div_i + 1], pred_truth[i, div_i + 1],
                                            lev_np[i, div_i + 1], p[i, div_i + 1], r[i, div_i + 1], ap50[i, div_i + 1],
                                            ap[i, div_i + 1]), logger)
                        print_log(pf % (
                            f'{names[c]}_over_{div_ang}', nt[i, -1], pred_truth[i, -1], lev_np[i, -1],
                            p[i, -1], r[i, -1], ap50[i, -1], ap[i, -1]), logger)

    print_log("###---------------------------------------------------------###", logger)
    return mp, mr, map50, map, ap, ap_class, p, r

def convert_small2big_json(jdict):
    for pic_info in jdict:
        file_name_split = pic_info['image_name'].split('-')
        if len(file_name_split) > 1:
            xxyy = file_name_split[-1].split("_")
            big_f_name = '-'.join(file_name_split[:-1])
        else:
            file_name_split = pic_info['image_name'].split('_')
            xxyy = file_name_split[-4:]
            big_f_name = '_'.join(file_name_split[:-4])

        try:
            x_bias, y_bias = int(xxyy[0]), int(xxyy[2])
            box = pic_info['bbox']
            box[0] += x_bias
            box[1] += y_bias
            pic_info['small_image_name'] = pic_info['image_name']
            pic_info['image_name'] = big_f_name
        except Exception as e:
            print('convert_small2big error', e)
