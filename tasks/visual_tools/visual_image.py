from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import print_log, visual_return, select_score_from_json
import os

if __name__ == '__main__':
    logger = None
    root_f = '/home/workspace/data/essential/merged_of_auto_resized_iou48_800_800dota_and_dior'
    last_conf = 0.4
    save_dir = 'results/val/merge/zjdet_neck_exp_best/exp3'

    img_file = f'{root_f}/labels/val.txt'
    anno_json = f'{root_f}/COCO/annotation/val.json'
    pred_json = f'{save_dir}/best_predictions.json'
    pred_json = select_score_from_json(pred_json,score_thresh=last_conf)
    image_file_id = {}
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)


    with open(img_file, 'r') as f:
        img0 = f.readline()
    img_prefix = os.path.split(img0)[0]
    print_log(
        f"##########################\nnow we collect and visual result with iou thresh of "
        f"0.5 and conf thresh of {last_conf}", logger=logger)

    visual_return(eval, anno, save_dir, img_prefix, class_area=None, score_thresh=last_conf,
                  logger=logger)