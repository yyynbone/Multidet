"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from models import attempt_load

FILE = Path(__file__).resolve()

from dataloader import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, Clahe
from utils import (set_logging,print_log, check_file, check_img_size, check_imshow,  colorstr, intersect_dicts,
                   increment_path, non_max_suppression_with_iof, print_args, scale_coords, strip_optimizer,
                   xyxy2xywh,Annotator, colors, save_one_box, select_device, time_sync, map_visualization)


@torch.no_grad()
def run(weights='checkpoints/zjs_s16.pt',  # model.pt path(s)
        cfg=None,
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        save_dir='results/detect/zjdet/exp',  # save results to project/name
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        is_bgr=1,
        last_conf=0.4,
        logger=None,
        model_class=None,
        **kwargs
        ):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download


    # Load model
    device = select_device(device)
    if model_class is not None:
        model = eval(model_class)()
        ck = torch.load(weights)
        csd = intersect_dicts(ck['state_dict'], model.state_dict(), key_match=False)
        model.load_state_dict(csd, strict=False)
        names = ck['names']
        model.to(device)
    else:
        model = attempt_load(weights, map_location=device, cfg=cfg, logger=logger)
        names =  {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

        model = model.module if hasattr(model, 'module') else model

    stride = int(max(model.stride)) if hasattr(model, 'stride') else 16

    imgsz = check_img_size(imgsz, s=stride)  # check image size


    if hasattr(model, 'model'):
        # Half
        half &= device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        model.model.half() if half else model.model.float()
    else:
        half = False

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        bs = len(dataset)  # batch_size
    else:
        if imgsz[0] == imgsz[1]:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False, is_bgr=is_bgr)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False, is_bgr=is_bgr)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if not is_bgr:
        ch_in = 1
    else:
        ch_in = 3
    warm_img = (1, ch_in, *imgsz)

    im = torch.zeros(*warm_img).to(device).type(torch.half if half else torch.float)  # input image
    model.forward(im)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    show_pre_process = False
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        if im.shape[0]!=ch_in:
            print(im.shape)
            im = im[:1, ...]  #20bridge 4airport
            im = im[:1, ...]*0.299  + im[1:2, ...]*0.587 + im[2:, ...]*0.114  # 22bridge 2airport
        if True:
            im = Clahe(im)
            show_pre_process = True
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32

        im /= 255  # 0 - 255 to 0.0 - 1.0
        # im = torch.tensor(im, dtype=torch.float32,device=device)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        print_log(im.shape, logger)

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        head_out = model(im, augment=augment, visualize=visualize)
        pred, da_seg_out = head_out[model.out_det_from]
        t3 = time_sync()
        dt[1] += t3 - t2
        # pred_j = {'pred':[pred_i.cpu().numpy().tolist() for pred_i in pred]}
        if len(pred[0])==1:
            pred_out = torch.where(pred > last_conf, torch.ones_like(pred), torch.zeros_like(pred))
            print_log(
                f"predict value is {pred[0][0]}  as {['background', 'object'][int(pred_out[0])]}",
                logger)
            dt[2] += time_sync() - t3
            seen+=1
            map_visualization(da_seg_out[0].sigmoid(), f_path=visualize+'/')
        else:
            # NMS
            # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred = non_max_suppression_with_iof(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, iof_nms=True)
            # pred_j['after_nms'] = [pred_i.cpu().numpy().tolist() for pred_i in pred]
            # with open('pred.json','a+') as f:
            #     json.dump(pred_j, f, indent=4)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    if show_pre_process:
                        im0 = (im[0].cpu().detach().clone().permute(1, 2, 0)*255).numpy().astype(int)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                # last conf filter
                det = det[det[:,4]>=last_conf]

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.1f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Print time (inference-only)
                print_log(f'{s}Done. ({t3 - t2:.3f}s)', logger)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print_log(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, ch_in, *imgsz)}' % t, logger)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print_log(f"Results saved to {colorstr('bold', save_dir)}{s}", logger)
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= 'checkpoints/zjs_s16.pt', help='model path(s)')
    parser.add_argument('--cfg', default=None, help='model yaml path(s)')
    parser.add_argument('--source', type=str, default= 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default= 'results/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--fuse', default=1, type=int, help='fuse or not')
    parser.add_argument('--last-conf', type=float, default=0.1, help='last conf thresh after nms')
    parser.add_argument('--is-bgr', default=1, type=int, help='input channel is 3 or not')
    parser.add_argument('--model-class', default=None, help='model class such as yolov5 yolov6 yolov8 etc')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # opt.imgsz = [960,544]
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    # Directories
    if 'weights' == opt.weights.split('/')[-2]:
        weight_file_name = '_'.join(opt.weights.split('/')[-4:-2]) + '_'
    else:
        weight_file_name = ''
    weight_file_name += str(Path(opt.weights).stem)
    # set the save dir name
    opt.save_dir = increment_path(Path(opt.project)  / weight_file_name / opt.name,
                                  exist_ok=opt.exist_ok)  # increment run
    (opt.save_dir / 'labels' if opt.save_txt else opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    opt.logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'val.log'))

    print_args(FILE.stem, opt, logger=opt.logger)
    print_log('#############################################', opt.logger)
    print_log(f'now we val {opt.weights}:\n saved in {opt.save_dir}', opt.logger)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
