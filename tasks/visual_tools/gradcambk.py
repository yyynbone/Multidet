import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
import cv2
from utils import non_max_suppression_with_iof
from models import attempt_load, Detect
from utils import xywh2xyxy, box_iou


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++
    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.
    Return:
        target_layer: found layer
    """
    print(model.model._modules)
    hierarchy = layer_name.split('_')
    target_layer = model.model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer

class YOLOV5GradCAM:
    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        # self.gradients = []
        # self.activations = []
        target_layer = find_yolo_layer(self.model, layer_name)
        print('Target_layer', target_layer)

        # Because of https://github.com/pytorch/pytorch/issues/61519,
        # we don't use backward hook to record gradients.
        target_layer.register_forward_hook(self.save_activation)  # self.save_activation
        target_layer.register_forward_hook(self.save_gradient)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))


    def save_activation(self, module, input, output):
        activation = output.clone()
        self.activations = activation.cpu().detach()
        # print('[INFO] saliency_map_size :', self.activations.shape[2:])
        # self.activations['value'] = activation.cpu().detach()
        # print('forward:', type(output))
        return None

    def save_gradient(self, module, input, output):
        # self.gradients = output.cpu().detach()
        # return None
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            self.gradients = output.clone().cpu().detach()
            return None

        #此时，若 if 中没跑，则开始注册backward hook, 注册方式为out.register_hook(function)
        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = grad.cpu().detach() # +1e-16

        output.register_hook(_store_grad)  #注册backward hook,
        return None

    # def forward(self, input_img, class_idx=True):
    #     """
    #     Args:
    #         input_img: input image with shape of (1, 3, H, W)
    #     Return:
    #         mask: saliency map of the same spatial dimension with input
    #         logit: model output
    #         preds: The object predictions
    #     """
    #     saliency_maps = []
    #     b, c, h, w = input_img.size()
    #     tic = time.time()
    #     head_out = model(input_img)  # forward
    #     det_preds = model.model[-1].test(head_out[-1])
    #     preds = non_max_suppression_with_iof(det_preds, 0.45, 0.3,  max_det=300,
    #                                         iof_nms=True)
    #     print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
    #     det_img = preds[0]
    #     # last conf filter
    #     det_img = det_img[det_img[:, 4] >= 0.4]
    #     for det in det_img:
    #
    #         *xyxy, conf, cls  = det
    #
    #         self.model.zero_grad()
    #         tic = time.time()
    #         conf.backward(retain_graph=True)
    #         print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
    #         gradients = self.gradients
    #         activations = self.activations
    #         b, k, u, v = gradients.size()
    #         alpha = gradients.view(b, k, -1).mean(2)
    #         weights = alpha.view(b, k, 1, 1)
    #         saliency_map = (weights * activations).sum(1, keepdim=True)
    #
    #         saliency_map = F.relu(saliency_map)
    #         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    #         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    #         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    #         saliency_maps.append(saliency_map)
    #     return saliency_maps, det_img
    #
    # def __call__(self, input_img):
    #     return self.forward(input_img)

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits = self.model(input_img)  # 调用forward_hook 相关函数，如之前定义的save_activation 和 save_gradient
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            tic = time.time()
            score.backward(retain_graph=True)  # 调用backward_hook 相关函数，如之前定义的_store_grad
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')

            activations = self.activations
            gradients = self.gradients
            if (gradients==0).all():
                print('error gard in score', score)
            else:
                print(gradients[0,0,0,:10], score)

            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)

            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        # self.gradients = []
        # self.activations = []
        return self.forward(input_img)

class YOLOV5TorchObjectDetector(nn.Module):
    def __init__(self,
                 model_weight,
                 device,
                 img_size,
                 names=None,
                 mode='eval',
                 confidence=0.4,
                 iou_thresh=0.45,
                 agnostic_nms=False):
        super(YOLOV5TorchObjectDetector, self).__init__()
        self.device = device
        self.model = None
        self.img_size = img_size
        self.mode = mode
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.model = attempt_load(model_weight, map_location=device)
        print("[INFO] Model is loaded")
        self.model.requires_grad_(True)
        self.model.to(device)
        if self.mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        # fetch the names
        if names is None:
            print('[INFO] fetching names from coco file')
            self.names = [ 'person', 'ddc', 'car', 'ddc', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        else:
            self.names = names

        # preventing cold start
        img = torch.zeros((1, 3, *self.img_size), device=device)
        self.model(img)

    @staticmethod
    def non_max_suppression(prediction, logits, conf_thres=0.6, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, labels=(), max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference and logits results
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls] and pruned input logits (n, number-classes)
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, 80), device='cpu')] * logits.shape[0]
        for xi, (x, log_) in enumerate(zip(prediction, logits)):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            log_ = log_[xc[xi]]
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
            # log_ *= x[:, 4:5]
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                # log_ = x[:, 5:]
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
                log_ = log_[conf.view(-1) > conf_thres]
            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

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
            logits_output[xi] = log_[i]
            assert log_[i].shape[0] == x[i].shape[0]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output, logits_output

    def forward(self, img):
        prediction = self.model(img, augment=False)[-1]
        if self.mode=='train':
            logits =  self.model.model[-1].get_logit(prediction)
            prediction = self.model.model[-1].test(prediction)
        else:
            prediction, log_pred = prediction
            logits = self.model.model[-1].get_logit(log_pred)

        prediction, logits = self.non_max_suppression(prediction, logits, self.confidence, self.iou_thresh,
                                                      classes=None,
                                                      agnostic=self.agnostic)
        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img.shape[0])] for _ in
                                                                        range(4)]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                for *xyxy, conf, cls in det:
                    # bbox = Box.box2box(xyxy,
                    #                    in_source=Box.BoxSource.Torch,
                    #                    to_source=Box.BoxSource.Numpy,
                    #                    return_int=True)
                    bbox = xyxy
                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    if self.names is not None:
                        self.class_names[i].append(self.names[cls])
                    else:
                        self.class_names[i].append(cls)
        return [self.boxes, self.classes, self.class_names, self.confidences], logits

def get_all_res_img(mask, res_img):

    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET).astype(np.float32)

    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, res_img

def get_roi_res_img(bbox, mask, res_img):

    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET).astype(np.float32)

    bbox = [int(b) for b in bbox]
    tmp = np.ones_like(res_img, dtype=np.float32) * 0
    tmp[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    heatmap = cv2.multiply(heatmap, tmp).astype(np.float32)

    res_img = cv2.add(res_img, heatmap)
    res_img = (res_img / res_img.max())
    return res_img, res_img

def put_text_box(bbox, cls_name, res_img, thickness=2):
    x1, y1, x2, y2 = [int(b) for b in bbox]
    res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    w, h = cv2.getTextSize(cls_name, 0, fontScale=thickness, thickness=2)[0]  # text width, height
    outside = y1 - h - 3 >= 0  # label fits outside box
    t0, t1 = x1, y1 - 3 if outside else y1 + h + 5
    res_img = cv2.putText(res_img, cls_name, (t0, t1), color=[255, 0, 0], fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          fontScale=1, thickness=1, lineType=cv2.LINE_AA)
    return res_img

def gradcam(gradients, activations, h, w):
    b, k, u, v = gradients.size()
    alpha = gradients.view(b, k, -1).mean(2)
    weights = alpha.view(b, k, 1, 1)
    saliency_map = (weights * activations).sum(1, keepdim=True)

    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    return saliency_map

class GradCAM:
    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = []
        self.activations = []

        # Because of https://github.com/pytorch/pytorch/issues/61519,
        # we don't use backward hook to record gradients.
        for layer in model.model.model:
            if isinstance(layer, Detect):

                for lm in layer.m:
                    lm.register_forward_hook(self.save_activation)  # self.save_activation
                    lm.register_forward_hook(self.save_gradient)
            # else:
            #     layer.register_forward_hook(self.save_activation)  # self.save_activation
            #     layer.register_forward_hook(self.save_gradient)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model(torch.zeros(1, 3, *img_size, device=device))


    def save_activation(self, module, input, output):
        activation = output.clone()
        self.activations.append(activation.cpu().detach())
        # print('[INFO] saliency_map_size :', self.activations.shape[2:])
        # self.activations['value'] = activation.cpu().detach()
        # print('forward:', type(output))
        return None

    def save_gradient(self, module, input, output):
        # self.gradients = output.cpu().detach()
        # return None
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            self.gradients.append(output.clone().cpu().detach())
            return None

        #此时，若 if 中没跑，则开始注册backward hook, 注册方式为out.register_hook(function)
        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients.append( grad.cpu().detach())

        output.register_hook(_store_grad)  #注册backward hook,
        return None

    # def forward(self, input_img, class_idx=True):
    #     """
    #     Args:
    #         input_img: input image with shape of (1, 3, H, W)
    #     Return:
    #         mask: saliency map of the same spatial dimension with input
    #         logit: model output
    #         preds: The object predictions
    #     """
    #     saliency_maps = []
    #     b, c, h, w = input_img.size()
    #     tic = time.time()
    #     head_out = model(input_img)  # forward
    #     det_preds = model.model[-1].test(head_out[-1])
    #     preds = non_max_suppression_with_iof(det_preds, 0.45, 0.3,  max_det=300,
    #                                         iof_nms=True)
    #     print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
    #     det_img = preds[0]
    #     # last conf filter
    #     det_img = det_img[det_img[:, 4] >= 0.4]
    #     for det in det_img:
    #
    #         *xyxy, conf, cls  = det
    #
    #         self.model.zero_grad()
    #         tic = time.time()
    #         conf.backward(retain_graph=True)
    #         print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
    #         gradients = self.gradients
    #         activations = self.activations
    #         b, k, u, v = gradients.size()
    #         alpha = gradients.view(b, k, -1).mean(2)
    #         weights = alpha.view(b, k, 1, 1)
    #         saliency_map = (weights * activations).sum(1, keepdim=True)
    #
    #         saliency_map = F.relu(saliency_map)
    #         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    #         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    #         saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    #         saliency_maps.append(saliency_map)
    #     return saliency_maps, det_img
    #
    # def __call__(self, input_img):
    #     return self.forward(input_img)

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_maps = []
        b, c, h, w = input_img.size()
        tic = time.time()
        preds, logits = self.model(input_img)  # 调用forward_hook 相关函数，如之前定义的save_activation 和 save_gradient
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for logit, cls, cls_name in zip(logits[0], preds[1][0], preds[2][0]):
            if class_idx:
                score = logit[cls]
            else:
                score = logit.max()
            self.model.zero_grad()
            self.gradients = []
            tic = time.time()
            score.backward(retain_graph=True)  # 调用backward_hook 相关函数，如之前定义的_store_grad
            print(f"[INFO] {cls_name}, model-backward took: ", round(time.time() - tic, 4), 'seconds')
            self.gradients = self.gradients[::-1]
            for activations, gradients in zip(self.activations, self.gradients):
                if (gradients == 0).all():
                    print('error gard in score', score)
                else:
                    print(gradients[0, 0, 0, :10], score)
                saliency_map = gradcam(gradients, activations, h, w)
                saliency_maps.append(saliency_map)
        return saliency_maps, logits, preds

    def __call__(self, input_img):
        self.gradients = []
        self.activations = []
        return self.forward(input_img)


if __name__ == '__main__':
    weight = '../../results/train/drone/zjdet_bocat/exp/weights/best.pt'
    device = 'cpu'
    img_path = '../../../data/visdrone/images/train/0000071_04085_d_0000007.jpg'
    img_size = (960, 540)
    im = torch.zeros(1, 3, img_size[1], img_size[0])
    bgr = True
    gramtype = 'all'
    if not bgr:
        img = cv2.imread(img_path, 0)
        im[0] = torch.tensor(img[:, :][None] / 255.)
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        im[0] = torch.tensor(img.transpose(2, 0, 1) / 255.)
    model = YOLOV5TorchObjectDetector(weight, device=device, img_size=img_size)
    saliency_method = YOLOV5GradCAM(model=model, layer_name='model_19_cv2', img_size=img_size) # 从model15 到16开始出现gradient.all()==0,
    # saliency_method = GradCAM(model=model, layer_name='',
    #                                 img_size=img_size)  # 从model15 到16开始出现gradient.all()==0,
    masks,  logits, [boxes, _, class_names, _]  = saliency_method(im)
    layer_num = len(masks)/len(logits[0])
    result = im.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    for i, mask in enumerate(masks):
        # if i<2:
        #     continue
        res_img = result.copy()
        # bbox, cls_name = pred_img[i][:4], pred_img[i][6]
        # bbox, cls_name = boxes[0][i], class_names[0][i]
        bbox, cls_name = boxes[0][int(i/layer_num)], class_names[0][int(i/layer_num)]
        if gramtype == "all":
            res_img, heatmat = get_all_res_img(mask, res_img)
            color_img = (res_img * 255).astype(np.uint8)

        else:
            res_img, heatmat = get_roi_res_img(bbox, mask, res_img)
            color_img = (res_img * 255).astype(np.uint8)

        color_img = put_text_box(bbox, cls_name, color_img)

        cv2.imshow('result', color_img)
        cv2.waitKey(0)  # 1 millisecond