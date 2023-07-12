import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
import math
import cv2
from utils import non_max_suppression_with_iof, xywh2xyxy, box_iou  #将plt.get_backend 修改为agg
from models import attempt_load, Detect #将plt.get_backend 修改为agg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# print(matplotlib.get_backend())
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

def feature_visualization(x):
    """
    x:              Features to be visualized
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(16, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 4), 4, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')
        plt.show()
        plt.close()


def plt_plot(x):
    plt.figure(figsize=(12, 9), tight_layout=True)
    plt.imshow(x)
    plt.show()
    plt.close()

def gradcam(gradients, activations, h, w):
    if activations.ndim==3:
        b, k, a = activations.size()
        scale_factor = a/h/w
        activations = activations.view(b, k, int(scale_factor*h), -1)
    # feature_visualization(activations)
    if gradients is not None:
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # saliency_map = F.leaky_relu(saliency_map)
    else:
        # saliency_map = activations.sigmoid().sum(1, keepdim=True)
        saliency_map = activations.sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
    # plt_plot(saliency_map.squeeze())
    saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
    # saliency_map = F.interpolate(saliency_map, size=(h, w), mode='nearest')
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    div_tensor = max(saliency_map_max - saliency_map_min, 1e-8)
    saliency_map = (saliency_map - saliency_map_min).div(div_tensor).data
    # plt_plot(saliency_map.squeeze())
    return saliency_map

def get_module(model,module_dict, prefix=''):
    prefix +=model._get_name()
    layer_num = 0
    for (k, lm) in model.named_children():
        layer_num+=1
        if len(lm._modules):
            module_dict = get_module(lm, module_dict, prefix+k)
        else:
            module_dict[prefix+k] = lm
    if layer_num==0:
        module_dict[prefix] =  model
    return module_dict

class GradCAM:
    def __init__(self, model, layer=None, img_size=(640, 640),use_grad=True):
        if use_grad:
            model.requires_grad_(True)  # 使用grad weight加权
        else:
            model.requires_grad_(False)  # 使用grad weight加权
        model.eval()
        self.use_grad = use_grad
        self.model = model
        self.gradients = []
        self.activations = []
        self.cam_layer = []
        model_layer = len(self.model.model)
        print('model layer length is :', model_layer)
        if layer is not None:
            if isinstance(layer, int):
                layer = (0, layer)

            if isinstance(layer, list):
                for i, l in enumerate(layer):
                    if l < 0:
                        layer[i] = model_layer + l

            elif isinstance(layer, tuple):
                start = layer[0]+model_layer if layer[0]<0 else layer[0]
                end = layer[1]+model_layer+1 if layer[1]<0 else layer[1]
                layer= range(start, end)
        else:
            layer = input('which layer do you want to select:(input the id number and split in space key,\n').split(' ')
            layer = [int(i) for i in layer]
        # Because of https://github.com/pytorch/pytorch/issues/61519,
        # we don't use backward hook to record gradients.
        module_dict = {}
        for i, layer_module in enumerate(self.model.model):
            if layer is not None:
                if i not in layer:
                    continue
            # if isinstance(layer_module, Detect):
            # if i==model_layer-1:
            #     for k, lm in layer_module._modules.items():
            #         lm.register_forward_hook(self.save_activation)  # self.save_activation
            #         lm.register_forward_hook(self.save_gradient)
            #         self.cam_layer.append(f'layer_{i}_{k}_'+ lm._get_name())
            # else:
            #     layer_module.register_forward_hook(self.save_activation)  # self.save_activation
            #     layer_module.register_forward_hook(self.save_gradient)
            #     self.cam_layer.append(f'layer_{i}_'+layer_module._get_name())
            module_dict = get_module(layer_module, module_dict)
        print(module_dict)
        for k, layer_module in module_dict.items():
            layer_module.register_forward_hook(self.save_activation)  # self.save_activation
            layer_module.register_forward_hook(self.save_gradient)
        self.cam_layer =  list(module_dict.keys())
        # device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        # self.model(torch.zeros(1, 3, *img_size, device=device))


    def save_activation(self, module, input, output):
        try:
            activation = output.clone()
        except:
            pass
        print(module, output.size())
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
            self.gradients.append(None)
            # self.gradients.append(torch.ones(output.size()))
            return None

        #此时，若 if 中没跑，则开始注册backward hook, 注册方式为out.register_hook(function)
        # Gradients are computed in reverse order
        def _store_grad(grad):
            gradient = grad.cpu().detach() # + 1e-16
            self.gradients.append( gradient)

        output.register_hook(_store_grad)  #注册backward hook,
        return None


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
        prediction = self.model(input_img, augment=False)[-1]
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        if self.model.training:
            logits = self.model.model[-1].get_logit(prediction)
            prediction = self.model.model[-1].test(prediction)

        else:
            prediction, log_pred = prediction
            logits = self.model.model[-1].get_logit(log_pred)

        if self.use_grad:

            # select_id = logits[..., 0].argsort(-1, descending=True)[:100]
            # iou_log = logits[:, select_id, 0].mean()

            # iou_log = logits[..., 0].flatten().sort(descending=True)[0]  #sort 返回 （value, key）
            # iou_log = iou_log.mean()

            # iou_log = logits[..., 0].flatten().max()  # sort 返回 （value, key）
            iou_log = logits[..., 0].flatten().mean()  # sort 返回 （value, key）
            print(iou_log)

            self.model.zero_grad()
            tic = time.time()
            iou_log.backward(retain_graph=True)
            self.gradients = self.gradients[::-1]
            print("[INFO] model-backward took: ", round(time.time() - tic, 4), 'seconds')

        preds = non_max_suppression_with_iof(prediction, 0.25, 0.6,  max_det=100,
                                            iof_nms=False)
        det_img = preds[0]
        # last conf filter
        det_img = det_img[det_img[:, 4] >= 0.4]  # [*xyxy, conf, cls ]

        # preds = prediction[0][prediction[0][...,4]>=0.5]
        #
        #
        # boxes = xywh2xyxy(preds[:, :4])
        # conf = preds[:,4:5]
        # cls_score, j = preds[:, 5:].max(1, keepdim=True)
        # det_img = torch.cat((boxes, conf, j.float()), 1) # [*xyxy, conf, cls ]

        for i,(activations, gradients) in enumerate(zip(self.activations, self.gradients)):
            if gradients is not None:
                if (gradients == 0).all():
                    print('error gard in layer', i)
                    gradients = None
                # else:
                #     print(gradients[0, 0, 0, :10])
            saliency_map = gradcam(gradients, activations, h, w)
            saliency_maps.append(saliency_map)
        return saliency_maps, det_img


    def __call__(self, input_img):
        self.gradients = []
        self.activations = []
        return self.forward(input_img)


if __name__ == '__main__':
    weight = '../../results/train/drone/zjdet_v8s/exp/weights/best.pt'
    device = 'cpu'
    img_path = '../../../data/visdrone/images/train/0000071_04085_d_0000007.jpg'
    img_size = (480, 270)
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

    model = attempt_load(weight, map_location=device)
    names = getattr(model, 'name', list(range(80)))
    print("[INFO] Model is loaded")

    model.to(device)
    # if mode=='train':
    #     model.train()
    #     model.requires_grad_(True)
    # else:
    #     model.requires_grad_(True)  # 使用grad weight加权
    #     model.eval()

    saliency_method = GradCAM(model=model, layer=[-2, -1],
                                    img_size=img_size, use_grad=True)
    masks,  det_img  = saliency_method(im)
    layer_num = len(masks)
    result = im.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    for i, mask in enumerate(masks):
        # if i<2:
        #     continue
        res_img = result.copy()

        if gramtype == "all":

            res_img, heatmat = get_all_res_img(mask, res_img)
            color_img = (res_img * 255).astype(np.uint8)
            for *xyxy, conf, cls in reversed(det_img):
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.1f}')
                color_img = put_text_box(xyxy, label, color_img)
            cv2.imshow(saliency_method.cam_layer[i], color_img)
            cv2.waitKey(0)  # 1 millisecond

        else:
            for *xyxy, conf, cls in reversed(det_img):
                res_img, heatmat = get_roi_res_img(xyxy, mask, res_img)
                color_img = (res_img * 255).astype(np.uint8)
                c = int(cls)  # integer class
                label = (f'{names[c]} {conf:.1f}')
                color_img = put_text_box(xyxy, label, color_img)
                cv2.imshow('result', color_img)
                cv2.waitKey(0)  # 1 millisecond