import argparse
import yaml
from copy import deepcopy
from pathlib import Path
# from models.head import Detect, YOLOv8Detect, IDetect
from models.head import *
from utils import set_logging, check_yaml, make_divisible, print_args
from models.layers import *

from dataloader.autoanchor import check_anchor_order
from utils import (feature_visualization, fuse_conv_and_bn, initialize_weights, intersect_dicts,
                   cal_flops, scale_img, select_device, time_sync, iter_extend_axis, print_log)

LOGGER = set_logging(__name__)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = False

det_head = (Detect, YOLOv8Detect, IDetect)
cls_head = (Classify, YOLOv8Classify, SqueezenetClassify, ObjClassify, Flatten)
def parse_model(d, ch_in, logger=LOGGER):  # model_dict, input_channels(3)
    print_log(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}", logger)
    anchors, nc, gd, gw = d.get('anchors', []), d['nc'], d.get('depth_multiple', 1), d.get('width_multiple', 1)
    backbone, head = d['backbone'], d['head']
    neck = d.get('neck', [])
    filter =  d.get('filter', [])
    neck_from = len(filter) + len(backbone)
    head_from = len(neck) + neck_from
    depth_layer = d.get('depth_layer', [])
    width_layer = d.get('width_layer', [])

    layers, save = [],  []  # layers, savelist

    for i, (f, m, args) in enumerate(filter + backbone  + neck + head):  # [from,  module, args], args=[ch_in , ch_out, etc]
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        if m in width_layer:

            if i == 0 or f == 'input':
                args.insert(0, ch_in)
                args[1] = make_divisible(args[1] * gw, 8)
            elif m in [str(task) for task in (*cls_head, *det_head)]:
                if isinstance(args[0], int):
                    args[0] = make_divisible(args[0] * gw, 8)
                else:
                    for ch_i, head_out_channel in enumerate(args[0]):
                        args[0][ch_i] = make_divisible(head_out_channel * gw, 8)
            else:
                # channel_in, channel_out = args[0], args[1]
                # if isinstance(channel_in, list):
                #     args[0] = 0
                #     for cha_in in channel_in:
                #         args[0]+= make_divisible(cha_in * gw, 8)
                # else:
                #     args[0] = make_divisible(channel_in * gw, 8)
                # args[1] = make_divisible(channel_out * gw, 8)
                channel_in, channel_out = args[0], args[1]
                if isinstance(channel_in, list):
                    for ch_i, cha_in in enumerate(args[0]):
                        args[0][ch_i]= make_divisible(cha_in * gw, 8)
                else:
                    args[0] = make_divisible(args[0] * gw, 8)
                args[1] = make_divisible(args[1] * gw, 8)

        if m in depth_layer:
            n = args[2]
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            args[2] = n

        m = eval(m) if isinstance(m, str) else m  # eval strings
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print_log(f'{i:>3}{str(f):>18}{np:10.0f}  {t:<40}{str(args):<30}', logger)  # print
        # if isinstance(f, str):
        #     print(f)
        #     print(f=='input')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1 and not isinstance(x, str))  # append to savelist
        layers.append(m_)
    return nn.Sequential(*layers), sorted(save), (neck_from, head_from)

def attempt_load(weights, map_location=None, cfg=None, inplace=True, fuse=True, ch=3, nc=80, logger=LOGGER):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        if cfg is None:
            ckpt_model = ckpt['ema' if ckpt.get('ema') else 'model']
        else:
            if ckpt.get( 'names'):
                nc = len(ckpt['names'])
            ckpt_model = Model(cfg, ch=ch, nc=nc).to(map_location)
            if ckpt.get('names'):
                ckpt_model.names = ckpt['names']
            csd = intersect_dicts(ckpt['state_dict'], ckpt_model.state_dict(), exclude=['anchor'])  # intersect
            ckpt_model.load_state_dict(csd, strict=False)
            if ckpt.get('hyp'):
                ckpt_model.hyp = ckpt['hyp']
            else:
                with open(Path(cfg).resolve().parents[1]/'hyp/hyp.yaml') as f:
                    ckpt_model.hyp = yaml.safe_load(f)
        ckpt_model.logger = logger
        if fuse:
            model.append(ckpt_model.float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt_model.float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, *det_head, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m)==Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print_log(f'Ensemble created with {weights}\n', logger)
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble

class Model(nn.Module):
    def __init__(self, cfg='zjdet.yaml', freeze=None, ch=3, nc=80, anchors=None, logger=LOGGER, verbose=False, imgsz=(640,640)):  # model, input channels, number of classes
        super().__init__()
        self.logger =  logger
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # self.det_out_idx = self.yaml.get('det_out_idx', -1)
        # self.seg_out_idx = self.yaml.get('seg_out_idx', None)
        # Define model
        ch = self.yaml['ch_input'] = self.yaml.get('ch_input', ch)  # input channels
        self.nc = self.yaml['nc'] = nc # self.yaml.get('nc', nc)
        if anchors and self.yaml.get('anchors'):
            print_log(f'Overriding model.yaml anchors with anchors={anchors}', self.logger)
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, (self.neck_from, self.head_from) = parse_model(deepcopy(self.yaml), ch, logger=logger)  # model, savelist
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        self.freeze = freeze
        self.class_conf = 0.4 #set the filter bg and obj conf

        # Build strides, anchors
        # m = self.model[-1]  # Detect()
        head = self.model[self.head_from: ]  # Detect()
        for i, m in enumerate(head):
            s = 256  # 2x min stride
            if isinstance(m, det_head):
                # ERROR, can't define the module here, it would collect det_head as a model to self, so self will be Model('model':xxx, 'det_from':xx)
                # so, when we calculate flops, it will calculate det_from again and again.
                # self.det_head = m
                self.det_head_idx = self.head_from + i
                m.inplace = self.inplace
                with torch.no_grad():
                    self.need_stride = True
                    model_out = self.forward(torch.zeros(1, ch, s, s))
                    detects = model_out[i]
                    m.stride = torch.tensor([s / x.shape[-2] for x in detects])

                # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
                if len(m.anchors):
                    m.anchors /= m.stride.view(-1, 1, 1)
                    check_anchor_order(m, self.logger)
                self.stride = m.stride
                if hasattr(m, 'bias_init'):
                    m.bias_init()
            # else:
            #     with torch.no_grad():
            #         model_out = self.forward(torch.zeros(1, ch, s, s))

        # Init weights, biases
        initialize_weights(self)
        self.info(verbose=verbose, img_size=imgsz)
        self.need_stride = False

    def forward(self, x, augment=False, profile=False, visualize=False):
        # if not hasattr(self, 'det_out_idx'):`
        #     self.det_out_idx = 24
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        head_out = [] # head output
        cache = [] # outputs
        dt = [] # date time
        thops = []
        paras = []
        self.out_det_from = 0
        self.filter_bool = None
        # print(f"image, {x[0, 0, :5, :5]}")
        input = x
        for i, m in enumerate(self.model):
            if not len(x):
                if isinstance(m, det_head):
                    cls_pred = [x for _ in range(m.nl)]
                    det_out = x.view(input.shape[0], 0, m.no)# modify det out, it should be .view(bs, -1, self.no)
                    head_out.append(cls_pred if self.training else (det_out, cls_pred))  # return (cls, det_out)
                continue
            # add freezed layer specially the layer of batched normal, because of requires_grad is inefficient to running_mean and var
            if isinstance(self.freeze, list):
                if f'model.{i}.' in self.freeze:
                    m.eval()
            if m.f != -1:  # if not from previous layer
                if m.f=='input':
                    if self.filter_bool is not None:
                        x = input[self.filter_bool]
                    else:
                        x = input
                else:
                    x = cache[m.f] if isinstance(m.f, int) else [x if j == -1 else cache[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt, thops, paras)
            # print(i, m)
            if not isinstance(m, cls_head):
                x = m(x)  # run
                # try:
                #     x = m(x)# run
                # except:
                #     print_log(f'{i}, is {m}', self.logger)
                #     print_log([j.shape for j in (x if isinstance(x, list) else [x])], self.logger)
            # if i==0:
            #     from utils import show_model_param
            #     show_model_param(m, path='./exp')
            #     print(f"{i}, {x[0, 0, :5, :5]}")
            if isinstance(m, cls_head):
                if i < self.head_from:
                    cls_out = m(x)
                    if not self.need_stride and not profile:
                        self.out_det_from = 1
                        out = cls_out.sigmoid() if m.training else cls_out[0]

                        if getattr(self, 'train_val_filter', True):
                            cls_conf  = getattr(self, 'class_conf', 0.4)
                            out_int = torch.where(out > cls_conf, torch.ones_like(out), torch.zeros_like(out))  # (n,1)
                            # out_bool = out_int.squeeze().bool()  # not squeeze,if n>1, this shape from (n,1) to (n); if n=1, to (), eg: torch.tensor([[1]]) to torch.tensor(1)
                            out_bool = out_int.flatten().bool()
                            x = x[out_bool] # RuntimeError: NYI: Named tensors are not supported with the tracer
                            # if not len(x):
                            #     print_log('return in classify, all background', self.logger)
                            #     cls_pred = [x for _ in range(self.nl)]
                            #     det_out = x.view(x.shape[0], 0, )# modify det out, it should be .view(bs, -1, self.no)
                            #     return (cls_out if m.training else (out_int, cls_out[1]),
                            #             cls_pred if self.training else (det_out, cls_pred))  # return (cls, det_out)
                            cache = [sx[out_bool] if sx is not None else None for sx in cache ]
                            self.filter_bool = out_bool
                        # modify: (not pred.sigmoid() but pred, a pred_out result of (n,1)
                        # head_out.append((out, out_bool))
                        head_out.append(cls_out if m.training else (out, cls_out[1]))
                else:
                    x = m(x)

            if i >= self.head_from:
                if self.filter_bool is not None:
                    x = iter_extend_axis(x, self.filter_bool)
                head_out.append(x)

            cache.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        return tuple(head_out)  # head_out is  a list to tuple, which maybe 1 or more head

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[self.det_out_idx].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt, thops, params):
        c = isinstance(m, det_head)  # is final layer, copy input as inplace fix

        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100) # so here we no need to plus 1000
        thops.append(o)
        params.append(m.np)
        if m == self.model[0]:
            print_log(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module':>40s}   {'input_size':>30s}", self.logger)
        print_log(f'{dt[-1]:10.2f} {o:10.4f} {m.np:10.0f}  {m.type:40s}  {[j.shape for j in x] if isinstance(x, list) else x.shape}', self.logger)
        if c or isinstance(m, cls_head):
            if isinstance(x, list):
                device = x[0].device
                ds = x[0].shape[0]
            else:
                device = x.device
                ds = x.shape[0]
            print_log(f"{sum(dt):10.2f} {sum(thops):>10.4f} {sum(params):>10.0f}  Total use {device} ", self.logger)
            print_log(f"{sum(dt)/ds:10.2f} {sum(thops)/ds:>10.4f} {sum(params):>10.0f}  Per image use {device}", self.logger)

    def _print_biases(self):
        m = self.model[self.det_head_idx]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print_log(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()), self.logger)

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print_log('Fusing layers... ', self.logger)
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info(verbose=True)
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        n_p = sum(x.numel() for x in self.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)  # number gradients
        if verbose:
            print_log(
                f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}", self.logger)
            for i, (name, p) in enumerate(self.model.named_parameters()):
                name = name.replace('module_list.', '')
                print_log('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                      (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()), self.logger)
        self.need_stride = True
        flops = cal_flops(self, img_size, verbose=verbose)
        self.need_stride = False
        layer_num = len(list(self.modules()))
        print_log(f"Model Summary: {layer_num}layers, {n_p} parameters, {n_g} gradients,image size is {img_size}, "
                         f"{flops:.1f} GFLOPs in {next(self.parameters()).device}", self.logger)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[self.det_head_idx] if hasattr(self, 'det_head_idx') else self.model[-1]
        # m = self.det_head
        if isinstance(m, det_head):
            m.stride = fn(m.stride)
            if hasattr(m, 'grid'):
                m.grid = list(map(fn, m.grid))
                if isinstance(m.anchor_grid, list):
                    m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    def save(self, file_name):
        torch.save(self, file_name)


if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=ROOT/'configs/model/zjdet_unet.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', default=True, help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML

    print_args(FILE.stem, opt)
    device = select_device(opt.device)
    ch_in = 3
    nc = 4
    input_shape = (ch_in, 544, 960)
    # Create model
    if check_yaml(opt.cfg):
        model = Model(opt.cfg, ch=ch_in, nc=nc, imgsz=(640, 640)).to(device)
        model.eval()
    else:
        weight = ROOT/ 'checkpoints/squeezenet.pt'
        model = attempt_load(weight, map_location=device, ch=ch_in, nc=1)


    # # Profile
    # if opt.profile:
    #     # img = torch.rand(32 if torch.cuda.is_available() else 1, ch_in, 320, 320).to(device)
    #     img = torch.rand(1, *(input_shape)).to(device)
    #     y = model(img, profile=True)
    #
    # # Test all models
    # if opt.test:
    #     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
    #         try:
    #             _ = Model(cfg)
    #         except Exception as e:
    #             print(f'Error in {cfg}: {e}')
    # torch.cuda.empty_cache()

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph



    from utils import get_model_complexity_info

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')   # GMac , 1 GMac = 2GFLOPs
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
