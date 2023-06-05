import os
import argparse
import yaml

from pathlib import Path

from glob import glob
from torch.utils.data import DataLoader

from utils import ROOT
from models import attempt_load
from loss import *
from dataloader import LoadImagesAndLabels, get_dataset, SuffleLoader
from tasks import val

from utils import ( set_logging, check_dataset, check_yaml, load_args, increment_path, select_device, print_args,print_log, select_class_tuple)
FILE = Path(__file__).resolve()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--data-prefile', type=str, default=ROOT / 'configs/preprocess/data_preprocess.yaml', help='data preprocess.yaml path')
    parser.add_argument('--weights', type=str,  nargs='+', default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', default=None, help='model yaml path(s)')
    parser.add_argument('--opt-file', type=str, default=None, help='opt file which load')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--bgr', type=int, default=1, help='if 1 bgr,0 gray')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'results/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--compute-loss', type=str, default=None, help='loss name used')
    parser.add_argument('--div-area', nargs='+', type=int, default=None, help='area of pixel to divide')
    parser.add_argument('--last-conf', nargs='+', type=float, default=0.1, help='last conf thresh after nms')
    parser.add_argument('--visual-matched', action='store_true', help='match the gt and pred and visual')
    parser.add_argument('--filter-str', type=str, default='', help='filter and select the image name with string')
    parser.add_argument('--ignore-bkg', action='store_true', help='filter and image of background')
    parser.add_argument('--loss-num', type=int, default=3, help='loss num of class , detect or seg')
    parser.add_argument('--train-val-filter', action='store_true', help='filter first use the classify head')
    parser.add_argument('--val-train', action='store_true', help='valuate the train dataset')
    opt = parser.parse_args()
    load_args(opt)
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    return opt


def main(opt):

    # Data
    data_file = Path(opt.data).stem
    opt.data = check_dataset(opt.data)  # check
    opt.data_prefile = check_yaml(opt.data_prefile)  # check YAML
    with open(opt.data_prefile, errors='ignore') as f:
        data_pres = yaml.safe_load(f)  # load hyps dict
    temp_weights = opt.weights
    weight_list = []
    # for temp_weight in temp_weights:
    #     if os.path.isdir(temp_weight):
    #         for weight in glob(f'{temp_weight}/*.pt'):
    #             # if 'zjs_' in weight and 'merge' not in weight:
    #             #     weight_list.append(weight)
    #             weight_list.append(weight)
    #     else:
    #         weight_list.append(opt.weights)
    for temp_weight in temp_weights:
        if os.path.isdir(temp_weight):
            weight_list += glob(f'{temp_weight}/**/*.pt', recursive=True)
        else:
            weight_list.append(temp_weight)

    for opt.weights in weight_list:
        opt.device = select_device(opt.device, batch_size=opt.batch_size)
        # Directories
        if 'weights' in opt.weights.split('/'):
            weight_file_name = '_'.join(opt.weights.split('/')[-4:-2]) + '_'
        else:
            weight_file_name = ''
        weight_file_name += str(Path(opt.weights).stem)

        opt.save_dir = increment_path(Path(opt.project)/ data_file / weight_file_name / opt.name, exist_ok=opt.exist_ok)  # increment run
        (opt.save_dir / 'labels' if opt.save_txt else opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        opt.logger = set_logging(name=FILE.stem, filename=Path(Path(opt.save_dir) / 'val.log'))

        print_args(FILE.stem, opt, logger=opt.logger)
        print_log(f'#############################################\nnow we val {opt.weights}:\n saved in {opt.save_dir}', opt.logger)

        # Load model
        opt.model = attempt_load(opt.weights, map_location=opt.device, cfg=opt.cfg, nc=int(opt.data['nc']), logger=opt.logger)
        opt.model = opt.model.module if hasattr(opt.model, 'module') else opt.model
        # for p in opt.model.parameters():
        #     p.requires_grad = False
        stride = int(opt.model.stride.max()) if hasattr(opt.model, 'stride') else 32
        opt.model.stride = stride
        opt.model.names = opt.data['names']  # get class names
        # if  not hasattr(opt.model, 'device'):
        opt.model.device = opt.device
        opt.model.train_val_filter = opt.train_val_filter
        # stride = 32  # grid size (max stride)
        # opt.imgsz = check_img_size(opt.imgsz, s=stride, logger=opt.logger)  # check image size

        if opt.compute_loss is not None:
            if isinstance(opt.compute_loss, str):
                opt.compute_loss = eval(opt.compute_loss)(opt.model, logger=opt.logger)


        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        if opt.val_train and task != 'train':
            tasks = (task, 'train')
        else:
            tasks = (task,)

        for opt.task in tasks:

            data_pre = data_pres[opt.task]
            task_data, data_pre = get_dataset(opt.data[opt.task], data_pre,  opt.imgsz, opt.batch_size, opt.logger,
                                          stride=int(stride),
                                          pad=.0,
                                          prefix=f'{opt.task}: ',
                                          bgr=opt.bgr,
                                          filter_str=opt.filter_str,
                                          filter_bkg=opt.ignore_bkg,
                                          select_class=select_class_tuple(opt.data))
            dataset = LoadImagesAndLabels(data_pre,
                                          task_data,
                                          logger=opt.logger,
                                          bkg_ratio=getattr(opt, "bkg_ratio", 0),
                                          obj_mask=getattr(opt,"val_obj_mask", 0),
                                          sample_portion=getattr(opt, "val_sample_portion", 1),
                                          pix_area=getattr(opt, 'pix_area', None)
                                          )

            dataset.cropped_imgsz = getattr(opt, "val_cropped_imgsz", False)
            # dataset.index_shuffle()
            batch_size = min(opt.batch_size, len(dataset))
            opt.dataloader = SuffleLoader(dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          sampler=None,
                          pin_memory=True,
                          collate_fn=LoadImagesAndLabels.collate_fn)
            opt.dataloader.shuffle_index(-1)

            # run normally

            results, maps,times = val(**vars(opt))
            msg = 'task{task}:   Loss({loss})\n' \
                  'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                  'Time: inference({t_inf:.4f}ms/frame)  nms({t_nms:.4f}ms/frame)'.format(
                task=opt.task, loss=results[-1],
                p=results[0], r=results[1], map50=results[2], map=results[3],
                t_inf=times[1], t_nms=times[2])
            print_log(msg, opt.logger)
    print_log('validate done', opt.logger)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
