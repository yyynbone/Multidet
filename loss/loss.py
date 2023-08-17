import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bbox_iou,  is_parallel, nancheck, infcheck, smooth_BCE, build_targets, TaskAlignedAssigner, \
    dist2bbox, make_anchors, bbox2dist, xywh2xyxy, print_log, xyxy2xywh, crop_mask
from loss.basic_loss import FocalLoss, CrossEntropyLoss

# class LossV5:
#     # Lossv5 losses
#     def __init__(self, model, autobalance=False, logger=None):
#         self.logger=logger
#         self.sort_obj_iou = False
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#
#         # Define criteria
#         # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
#         self.hyp = h
#         no_dist_model = model.module if is_parallel(model) else model
#         self.freeze = getattr(no_dist_model, "freeze", None)
#
#         self.det_loss = True
#         if self.freeze:
#             if any(str(no_dist_model.head_from) in x for x in self.freeze): # det and encoder is freeze
#                 self.det_loss = False
#
#         # class loss criteria
#         self.use_CE = h.get('CE', False)
#         if self.det_loss:
#             if not self.use_CE:
#                 BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']]).to(device))
#             else:
#                 BCEcls = CrossEntropyLoss(class_weight=h['cls_weight']).to(device)
#             BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
#
#             self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#
#             # Focal loss
#             g = h['fl_gamma']  # focal loss gamma
#             if g > 0:
#                 BCEobj = FocalLoss(BCEobj, g)
#                 if not self.use_CE:
#                     BCEcls = FocalLoss(BCEcls, g)
#
#             det = no_dist_model.model[no_dist_model.det_head_idx]  # Detect() module
#
#             self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#             self.balance =  h.get('obj_level_weight', self.balance)
#             self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#
#             self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
#             for k in 'nc', 'anchors':
#                 setattr(self, k, getattr(det, k))
#             self.loss_num = 3
#
#     def __call__(self, p, targets):  # predictions, targets, model
#         if isinstance(targets, (list, tuple)):
#             class_label, targets, seg_img = targets
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         cls_ls, iou_ls = torch.zeros(1, device=device), torch.zeros(1, device=device)
#         if self.det_loss:
#             tcls, tbox, indices, anchors = build_targets(p, targets, self.anchors, self.hyp['anchor_t'],)  # targets
#             # Losses
#             for i, pi in enumerate(p):  # layer index, layer predictions
#                 b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#                 tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#                 n = b.shape[0]  # number of targets
#                 if n:
#                     ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                     # Regression
#                     pxy = ps[:, :2].sigmoid() * 2 - 0.5
#                     pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                     pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                     iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                     iou_ls = (1.0 - iou).mean()  # iou loss
#                     lbox += iou_ls
#
#                     # Objectness
#                     score_iou = iou.detach().clamp(0).type(tobj.dtype)
#                     if self.sort_obj_iou:
#                         sort_id = torch.argsort(score_iou)
#                         b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
#                     tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
#
#                     if self.nc > 1:  # cls loss (only if multiple classes)
#                         if not self.use_CE:
#                             t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                             t[range(n), tcls[i]] = self.cp
#                             cls_ls = self.BCEcls(ps[:, 5:], t)
#
#                             # # BCE   RuntimeError: Expected object of scalar type Long but got scalar type Float
#                             # # for argument #2 'target' in call to _thnn_nll_loss_forward
#                             # lcls += BCEcls(ps[:, 5:], t.long())  # BCE
#                         else:
#                             cls_ls = self.BCEcls(ps[:, 5:], tcls[i])  # Cross Entropy
#                         lcls += cls_ls
#
#
#
#                 obj_ls= self.BCEobj(pi[..., 4], tobj)
#                 if obj_ls > 2:
#                     print_log(f"now in level {i} object loss bigger, which may cause erupt", self.logger)
#                 if nancheck([cls_ls, iou_ls, obj_ls]) or infcheck([cls_ls, iou_ls, obj_ls]):
#                     print_log(f"warning now in level {i},  we found a nan or inf in loss {[cls_ls, iou_ls, obj_ls]}", self.logger)
#
#                 lobj += obj_ls * self.balance[i]  # obj loss
#                 if self.autobalance:
#                     self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obj_ls.detach().item()
#
#             if self.autobalance:
#                 self.balance = [x / self.balance[self.ssi] for x in self.balance]
#             # print(lbox, lobj, lcls)
#             lbox *= self.hyp['box'] #
#             lobj *= self.hyp['obj'] # score
#             lcls *= self.hyp['cls'] # class
#             bs = tobj.shape[0]  # batch size
#             # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
#         else:
#             bs = p.shape[0]
#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
#

class LossV5:
    # Lossv5 losses
    def __init__(self, model, autobalance=False, logger=None):
        self.logger=logger
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters
        autobalance = h.get('autobalance', False)

        # Define criteria
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  modify
        self.hyp = h
        no_dist_model = model.module if is_parallel(model) else model
        self.freeze = getattr(no_dist_model, "freeze", None)

        self.det_loss = True
        if self.freeze:
            if any(str(no_dist_model.head_from) in x for x in self.freeze): # det and encoder is freeze
                self.det_loss = False

        # class loss criteria
        self.use_CE = h.get('CE', False)
        if self.det_loss:
            if not self.use_CE:
                BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']]).to(device))
            else:
                BCEcls = CrossEntropyLoss(class_weight=h['cls_weight']).to(device)
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

            self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

            # Focal loss
            g = h['fl_gamma']  # focal loss gamma
            if g > 0:
                BCEobj = FocalLoss(BCEobj, g)
                if not self.use_CE:
                    BCEcls = FocalLoss(BCEcls, g)

            det = no_dist_model.model[no_dist_model.det_head_idx]  # Detect() module

            self.obj_balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
            self.obj_balance =  h.get('obj_level_weight', self.obj_balance)
            self.cls_balance = h.get('cls_level_weight', [1., 1., 1.])
            self.box_balance = h.get('box_level_weight', [1., 1., 1.])
            print_log(f"cls box and obj balance is {self.cls_balance}, {self.box_balance}, {self.obj_balance}",
                      logger)
            self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index

            self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance
            for k in 'nc', 'anchors':
                setattr(self, k, getattr(det, k))
            self.loss_num = 3

    def __call__(self, p, targets):  # predictions, targets, model
        if isinstance(targets, (list, tuple)):
            class_label, targets, seg_img = targets
        device = targets.device
        loss = torch.zeros(3, device=self.device)  # box, obj, cls
        origin_loss = loss.detach()
        llcls, llbox, llobj = [], [], []
        cls_ls, iou_ls = torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs = p.shape[0]  # batch size
        if self.det_loss:
            tcls, tbox, indices, anchors = build_targets(p, targets, self.anchors, self.hyp['anchor_t'],)  # targets
            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2 - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    iou_ls = (1.0 - iou).mean()  # iou loss
                    llbox.append(iou_ls)

                    # Objectness
                    score_iou = iou.detach().clamp(0).type(tobj.dtype)
                    if self.sort_obj_iou:
                        sort_id = torch.argsort(score_iou)
                        b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                    if self.nc > 1:  # cls loss (only if multiple classes)
                        if not self.use_CE:
                            t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                            t[range(n), tcls[i]] = self.cp
                            cls_ls = self.BCEcls(ps[:, 5:], t)

                            # # BCE   RuntimeError: Expected object of scalar type Long but got scalar type Float
                            # # for argument #2 'target' in call to _thnn_nll_loss_forward
                            # lcls += BCEcls(ps[:, 5:], t.long())  # BCE
                        else:
                            cls_ls = self.BCEcls(ps[:, 5:], tcls[i])  # Cross Entropy
                        llcls.append(cls_ls)



                obj_ls = self.BCEobj(pi[..., 4], tobj)
                if obj_ls > 0.5:
                    print_log(f"now in level {i} object loss bigger, which may cause erupt", self.logger)
                #     self.obj_balance[i]*=2
                # else:
                #     self.obj_balance[i]=[4.0, 1.0, 0.4][i]

                if nancheck([cls_ls, iou_ls, obj_ls]) or infcheck([cls_ls, iou_ls, obj_ls]):
                    print_log(f"warning now in level {i},  we found a nan or inf in loss {[cls_ls, iou_ls, obj_ls]}", self.logger)

                llobj.append(obj_ls * self.obj_balance[i])  # obj loss
                if self.autobalance:
                    self.obj_balance[i] = self.obj_balance[i] * 0.9999 + 0.0001 / obj_ls.detach().item()

            if self.autobalance:
                self.obj_balance = [x / self.obj_balance[self.ssi] for x in self.obj_balance]

            # print(llcls, llbox, llobj)
            loss[0] = sum([l * w for l, w in zip(llbox, self.box_balance)]) #
            loss[1] = sum(llobj)  # score
            loss[2] = sum([l * w for l, w in zip(llcls, self.cls_balance)]) # class # is a constant not an array
            origin_loss = loss.clone().detach()
            loss[0] *= self.hyp['box']  # box gain
            loss[1] *= self.hyp['obj']  # obj gain
            loss[2] *= self.hyp['cls']  # cls gain

            # loss[0] = sum([l*w for l, w in zip(llbox, self.box_balance)]) * self.hyp['box']  #
            # loss[1] = sum(llobj) * self.hyp['obj'] # score
            # loss[2] = sum([l*w for l, w in zip(llcls, self.cls_balance)])* self.hyp['cls'] # class # is a constant not an array
            # print(lbox, lobj, lcls)
        return loss.sum() * bs, origin_loss  # loss(box, obj, cls)

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask)
        iou = bbox_iou(pred_bboxes[fg_mask].T, target_bboxes[fg_mask], x1y1x2y2=True, CIoU=True).T #输出[n]
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight.unsqueeze(-1)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


# Criterion class for computing training losses
class LossV8:

    def __init__(self, model, logger=None):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = model.module.model[model.module.det_head_idx] if is_parallel(model) else model.model[
            model.det_head_idx]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.loss_num = 3

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        if isinstance(targets, (list, tuple)):
            class_label, targets, seg_img = targets
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        origin_loss = loss.clone().detach()
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain

        return loss.sum() * batch_size, origin_loss  # loss(box, cls, dfl)

# Criterion class for computing training losses
class LossV8seg:

    def __init__(self, model, logger=None, overlap=True):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = model.module.model[model.module.det_head_idx] if is_parallel(model) else model.model[
            model.det_head_idx]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.loss_num = 4
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = overlap

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, targets):
        if isinstance(targets, (list, tuple)):
            class_label, targets, seg_img = targets
        loss = torch.zeros(self.loss_num, device=self.device)  # box, cls, dfl
        if self.loss_num==3:
            feats = preds[1] if isinstance(preds, tuple) else preds
            pred_masks, proto =  None, None
        else:
            feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        if pred_masks is not None:
            pred_masks = pred_masks.permute(0, 2, 1).contiguous()
            masks = seg_img.to(self.device).float()
            batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            if pred_masks is not None:
                for i in range(batch_size):
                    if fg_mask[i].sum():
                        mask_idx = target_gt_idx[i][fg_mask[i]]
                        if self.overlap:
                            gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                        else:
                            gt_mask = masks[i][mask_idx]
                        xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                        marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                        mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                        loss[3] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy,
                                                         marea)  # seg loss
        origin_loss = loss.clone().detach()
        loss[0] *= self.hyp['box']  # box gain
        loss[1] *= self.hyp['cls']  # cls gain
        loss[2] *= self.hyp['dfl']  # dfl gain

        if pred_masks is not None:
            loss[3] *= self.hyp['box'] / batch_size  # seg gain

        return loss.sum() * batch_size, origin_loss  # loss(box, cls, dfl)
    
    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        # Mask loss for one image
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()
    
class ClassifyLoss():
    def __init__(self,model, logger=None):
        use_BCE = model.hyp.get('use_BCE', True)
        use_focal = model.hyp.get('use_focal', False)

        self.device = next(model.parameters()).device  # get model device
        self.pos_weight = model.hyp.get('pos_weight', None)
        self.pos_gain = model.hyp.get('pos_gain', 1)
        print_log(f'ClassifyLoss bg_obj_weight is {self.pos_weight}', logger)
        if not use_BCE and self.pos_weight is not None:
            class_weight = [1, self.pos_weight]
        else:
            class_weight = None
        self.loss = CrossEntropyLoss(use_sigmoid=use_BCE, use_focal=use_focal, class_weight=class_weight).to(self.device)
        self.loss_num = 1
        self.label_pos_weight = model.hyp.get('max_pos_weight', 10)
        self.loss_style = model.hyp.get('loss_style', 1)


    def __call__(self, preds, targets):
        if isinstance(targets, (list, tuple)):
            class_label, targets, seg_img = targets
        else:
            class_label = targets

        if self.pos_weight in [0, 'None', None]:
            ft_num = class_label.shape[0] - class_label.sum().item()   #int(tensor) or tensor.item() if tensor is a constant int
            self.label_pos_weight = round(ft_num / max(class_label.sum().item(), 1), 2)  # torch.div(ft_num, max(class_label.sum(), 1))
        else:
            self.label_pos_weight = self.pos_weight
        # print(preds.shape, preds.dtype,  class_label.shape, class_label.dtype)
        cls_ls = self.loss(preds, class_label, pos_weight=self.pos_gain * torch.tensor([self.label_pos_weight], device=self.device), loss_style=self.loss_style)
        return cls_ls*len(preds), cls_ls.detach()


class ObjClassifyLoss():
    def __init__(self,model, logger=None):
        use_BCE = model.hyp.get('use_BCE', True)
        use_focal = model.hyp.get('use_focal', False)
        class_weight = model.hyp.get('cls_weight', None)

        self.device = next(model.parameters()).device  # get model device
        self.pos_weight = model.hyp.get('pos_weight', 10)
        self.pos_gain = model.hyp.get('pos_gain', 1)
        print_log(f'ClassifyLoss bg_obj_weight is {self.pos_weight}', logger)
        self.loss = CrossEntropyLoss(use_sigmoid=use_BCE, use_focal=use_focal, class_weight=class_weight).to(self.device)
        self.loss_num = 1
        self.loss_style = model.hyp.get('loss_style', 1)
        self.label_pos_weight =  10

    def __call__(self, preds, targets):
        if isinstance(targets, (list, tuple)):
            class_label, targets, seg_img = targets
        tobj = self.map(targets, preds)
        bs = tobj.shape[0]
        cls_ls = self.loss(preds.reshape(bs, -1), tobj.reshape(bs, -1), pos_weight=self.pos_gain * torch.tensor([self.pos_weight], device=self.device), loss_style=self.loss_style)
        return cls_ls*len(preds), cls_ls.detach()
    def map(self, targets, preds):
        """
        :param targets: (img_id, cls, x, y ,w, h) / imgsize
        :param preds: (b, c2, fy, fx)
        :return:
        """
        my, mx = preds.shape[2:4]
        targets[..., 2:] =  xywh2xyxy(targets[..., 2:]) * torch.tensor([mx, my, mx, my], device=targets.device)# gx1,gy1,gx2,gy2
        tobj = torch.zeros_like(preds, device=preds.device)  # target obj
        for img_id, cls, gx1, gy1, gx2, gy2  in targets.long():
            gx2 = min(max(gx1, gx2)+1, mx)
            gy2 = min(max(gy1, gy2) + 1, my)
            tobj[img_id, cls, gy1:gy2, gx1:gx2] = 1.
        return tobj


class TeacherLoss:
    def __init__(self, model, logger=None):
        super().__init__(model, logger)

    def __call__(self, p, targets):
        pass


def teacher_loss(student_out, teacher_out, temperature):
    """
    use the prediction of student and teacher, can be origin out or softmax or sigmoid out
    :param student_out:
    :param teacher_out:
    :param temperature:
    :return:
    """
    loss = torch.zeros(1, device=student_out.device)
    if isinstance(student_out, list):
        for (s_out, t_out) in zip(student_out, teacher_out):
            loss += -torch.sum(s_out.sigmoid()/temperature.log() * t_out.sigmoid()/temperature)
    return loss/student_out.shape[0]

