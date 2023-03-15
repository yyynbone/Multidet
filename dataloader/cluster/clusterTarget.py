import numpy as np
import torch
import os
import cv2
import random

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_iou(box1, box2, lamda=2):
    """
    :param box1: array [x1, y1, x2, y2]
    :param box2: array [x1, y1, x2, y2]
    :param lamda:放大比例
    :return: iou
    """
    def enlarge(x1, y1, x2, y2, lamda, img_size=832):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        w = lamda * w
        h = lamda * h

        x1 = x - w / 2 if (x - w / 2) > 0 else 0
        x2 = x + w / 2 if (x + w / 2) < img_size else img_size
        y1 = y - h / 2 if (y - h / 2) > 0 else 0
        y2 = y + h / 2 if (y + h / 2) < img_size else img_size

        return x1, y1, x2, y2

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # print(b1_x1, b1_y1, b1_x2, b1_y2)
    # print(b2_x1, b2_y1, b2_x2, b2_y2)
    b1_x1, b1_y1, b1_x2, b1_y2 = enlarge(b1_x1, b1_y1, b1_x2, b1_y2, lamda=lamda, img_size=800)
    b2_x1, b2_y1, b2_x2, b2_y2 = enlarge(b2_x1, b2_y1, b2_x2, b2_y2, lamda=lamda, img_size=800)

    w = min(b1_x2, b2_x2) - max(b1_x1, b2_x1) if (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) > 0 else 0
    h = min(b1_y2, b2_y2) - max(b1_y1, b2_y1) if (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)) > 0 else 0

    inter = w * h

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter

    iou = inter / (union + 0.00000001)
    # print(iou)
    return iou

def box_iou(box1, box2):
    """
    计算iou值
    :param box1: Tensor[N, 4] [x1, y1, x2, y2] 左上角 右下角
    :param box2: Tensor[M, 4] [x1, y1, x2, y2]
    :return: Tensor[N, M] 成对的boxes1和boxes2中每个元素的IoU值
    """
    # box1 = torch.Tensor(box1)
    # box2 = torch.Tensor(box2)
    def box_area(box):
        # 左上角 右下角格式
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def bdscan(Data, iou, MinPts, n):
    """
    :param Data: list [[x1, y1, x2, y2], ...]
    :param iou: iou阈值
    :param MinPts: 相交的个数阈值
    :param n: 成为一个集群的个数阈值
    :return: C 每个框属于那个簇， cluster：符合集群的框 list[[]]
    """
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    cluster = []  # 保存找到的集群
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        # print(data[p])
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (bbox_iou(Data[i], Data[p]) >= iou):  # and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k + 1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        # 这里可以换成iou
                        if (bbox_iou(Data[j], Data[pi]) >= iou):  # and (j!=pi):
                            M.append(j)
                    if len(M) >= MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1

        #  这里可以修改簇中个数的阈值
        if len(N) > n:
            cluster.append([i for i in N])
        # print('k:{}, num:{}'.format(k, len(N)))
        # print([i + 4 for i in N])
        # print("=========================================================")

    return C, cluster

def match(pred, gt, iou_thres=0.3):
    # pred = pred[pred[:, 4] > conf]  # 保留置信度大于阈值的框
    # gt_classes = torch.Tensor(gt[:, 0]).int()
    # detection_classes = torch.Tensor(pred[:, 5]).int()
    # 计算检测的框和真实框之间的iou
    # gt_boxes = xywh2xyxy(gt[:, 1:5])
    iou = box_iou(gt[:, 1:], pred[:, :4])
    # 保留iou大于阈值的框 预测的第一个框和真实的所有框的iou
    x = torch.where(iou > iou_thres)
    # iou大于阈值的个数
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
    if x[0].shape[0] > 1:
        # 按照第二列的大小进行排序，即按照iou从大到小排序
        matches = matches[matches[:, 2].argsort()[::-1]]
        # 第1列去重
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        # 按照第二列的大小进行排序，即按照iou从大到小排序
        matches = matches[matches[:, 2].argsort()[::-1]]
        # 第0列去重
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    return matches

def cluster_target(pred, gt, n=3):
    """
    :param pred: array[n1, 6],[x1, y1, x2, y2, conf, class]
    :param pred: array[k1, 5], [class, x, y, w, h],yolo的gt都是这个格式
    :return: 新的pred
    """
    gt[:, 1:5] = xywh2xyxy(gt[:, 1:5])
    # 真实类别标签
    gt_class = gt[:, 0]
    # print(gt_class)
    # 有多少类别，各个类别的数量
    gt_unique_classes, gt_n = np.unique(gt_class, return_counts=True)
    gt_nt = np.zeros((9,))

    for i in range(len(gt_unique_classes)):
        gt_nt[int(gt_unique_classes[i])] = gt_n[i]
    # num_class = gt_unique_classes.shape[0]
    # print("gt  ", gt_unique_classes, gt_n, gt_nt)
    # 预测类别标签
    pred_class = pred[:, 5]
    # 有多少类别，各个类别的数量
    pred_unique_classes, pred_n = np.unique(pred_class, return_counts=True)
    pred_nt = np.zeros((9,))

    for i in range(len(pred_unique_classes)):
        pred_nt[int(pred_unique_classes[i])] = pred_n[i]
    # print("pred:", pred_unique_classes, pred_nt)
    # print("pred  ", pred_unique_classes, pred_n, pred_nt)
    matches = match(torch.Tensor(pred), torch.Tensor(gt))
    # print(matches)
    # 将每个类别取出来 然后进行聚类
    for label in gt_unique_classes:
        get_gt = gt[:, 0] == label
        # 提取同一类别的原始idx
        get_idx = torch.where(torch.Tensor(get_gt)==True)[0].cpu().numpy().tolist()
        # print(get_idx)
        # print(sum(get_gt))
        # 做一个判断  当前样本中某个类别的数量大于阈值 进行聚类
        if sum(get_gt) > n:
            print(gt[get_gt])
            # 找到集群目标的索引 cluster中对应提取某一类别后框的相对idx  用相对idx找原始idx
            _, cluster = bdscan(gt[get_gt][:, 1:], 0.3, 3, 3)
            # print(cluster)
            for j in range(len(cluster)):
                add = []
                # 当前簇中有多少个目标
                num_cluster = len(cluster[j])
                #  拿到相对idx
                for idx in sorted(cluster[j]):
                    # print(get_idx[idx])
                    #  get_idx[idx] 原始idx 原始idx的框有没有被正确的检测到
                    count = matches[:, 0] == get_idx[idx]
                    # print(sum(count))
                    # 该集群中的这个框没有被检测到， 则添加到add中
                    if sum(count) == 0:
                        add.append(gt[get_gt][idx, :])
                # 未被检测到的数量 检测到的框超过一半  则认为当前集群全部检测到， 所以见没有检测到的框加到pred中
                num_not_detected = len(add)
                # 检测到的框超过一半  则认为当前集群全部检测到， 所以将没有检测到的框加到pred中 或者检测到多少个则可以代替也行
                if (num_cluster - num_not_detected) / num_cluster >= 0.5:
                    add = np.array(add).reshape(-1, 5)
                    # confidence置1，加入到pred中，匹配时优先使用加入的框
                    tmp = np.ones([num_not_detected, 1])
                    tmp = np.hstack((add[:, 1:], tmp, add[:, :1]))
                    # 将集群目标中未检测到的目标添加到预测框中
                    pred = np.vstack((pred, tmp))
    # gt[:, 1:5] = xyxy2xywh(gt[:, 1:5])

    return torch.Tensor(pred)