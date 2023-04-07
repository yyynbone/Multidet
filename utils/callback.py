from torch.utils.tensorboard import SummaryWriter
import warnings
import torch
from threading import Thread
from utils.logger import colorstr, print_log
from utils.torch_utils import de_parallel
from utils.plots import plot_images, plot_results

class Callback():
    def __init__(self, save_dir=None, opt=None, hyp=None, logger=None, include=('csv', 'tb')):
        self.save_dir = save_dir
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include

        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
        for k in self.include:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            print_log(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/", self.logger)
            self.tb = SummaryWriter(str(s))

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        # paths = self.save_dir.glob('*labels*.jpg')  # training labels
        pass


    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots, sync_bn):
        # Callback runs on train batch end
        if isinstance(targets, (tuple, list)):
            labels = targets[1]
        else:
            labels = targets
        if plots:
            if ni == 0:
                if not sync_bn:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # suppress jit trace warning
                        # if error Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions;
                        # which means the output  of model is a list, so you can tuple it.
                        try:
                            self.tb.add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
                        except:
                            print("jit trace wrong")
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, labels, paths, f), daemon=True).start()


    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        pass

    def on_val_end(self):
        # Callback runs on val end
        pass

    def on_fit_epoch_end(self, vals, epoch):
        # Callback runs at the end of each fit (train+val) epoch
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)



    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        pass

    def on_train_end(self, plots, epoch):
        # Callback runs on training end
        if plots:
            plot_results(file=self.save_dir / 'results.csv')  # save results.png
        files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        if self.tb:
            import cv2
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')

