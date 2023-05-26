import numpy as np
import torch

def labels_to_class_weights(labels, nc=80, empty_bins=1e16):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return np.array([])

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int8)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = empty_bins  # replace empty bins with 10**16
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights

# def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
#     # Rescale coords (xyxy) from img1_shape to img0_shape
#     if ratio_pad is None:  # calculate from img0_shape
#         gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#         pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#     else:
#         gain = ratio_pad[0][0]
#         pad = ratio_pad[1]
#
#     coords[:, [0, 2]] -= pad[0]  # x padding
#     coords[:, [1, 3]] -= pad[1]  # y padding
#     coords[:, :4] /= gain
#     clip_coords(coords, img0_shape)
#     return coords
def scale_coords(img1_shape, coords, img0_shape, resize_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if resize_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = resize_pad[0][0]/img0_shape[0]
        pad = resize_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xywh2xyxy(x, w=1, h=1, padw=0, padh=0):
    """
    Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def clip_label(x, max_x, max_y,  min_x=0,  min_y=0):
    np.clip(x[..., 0::2], min_x, max_x, out=x[..., 0::2])  # clip when using random_perspective()
    np.clip(x[..., 1::2], min_y, max_y, out=x[..., 1::2])  # clip when using random_perspective()

def xyxy2xywh(x, w=1, h=1, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    if not len(x):  # x为[]
        return x
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0::2] = w * x[:, 0::2] + padw  # top left x
    y[:, 1::2] = h * x[:, 1::2] + padh  # top left y
    return y

def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy

def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh

def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments

def build_targets(p, targets, level_anchors, anchor_t):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    nl, na = level_anchors.shape[:2]  # number of detection layers, number of anchors
    nt =  targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # cat targets and layer index(anchor index)  shape from (nt,6)->(3,nt,6)->(3,nt,7)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(nl):
        anchors = level_anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain  featmap_size like 20,40,or80

        # Match targets to anchors
        t = targets * gain   #size is (3,nt,7)
        if nt:
            # Matches
            #gt 与 anchor 的比值，筛选并限制最大倍数为4倍
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1 / r).max(2)[0] < anchor_t  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter, now t size is (1312,7) filtered number is 1312

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))  #size is (5,1312)
            t = t.repeat((5, 1, 1))[j]   # now filter again, size is (3909,7),so 3909 = 1312+ (1312-(the grid x bias=0.5)) + (1312-(the grid y bias=0.5))
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch

def select_class_tuple(data_dict):
    """
    get class id from old to new
    :param data_dict: dataset dictionary
    :return: tuple or list
    """
    # dataset class eg: ['vehicle', 'bridge', 'ship','airport', 'harbor', 'airplane', 'helipad', 'helicopter']
    if data_dict.get('xml_names'):
        return data_dict['xml_names']
    data_class = data_dict.get('data_names', data_dict['names'])
    #  class we only want  eg:['airplane']
    select_class = data_dict['names']
    se_id = [ data_class.index(sc) if sc in data_class else -1  for sc in select_class]
    # print(se_id)
    return tuple(se_id)