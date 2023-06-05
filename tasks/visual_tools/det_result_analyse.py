import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def det_result_analyse(gt, predictions, iou_type='bbox', iou=None, area_range="all", max_det=100):
    """
    iou_type: ["bbox"]
    iou: [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    area_range: ["all", "small", "medium", "large"], small:[0, 32**2], medium:[32**2, 96**2], large:[96**2, infinite]
    max_det: [1, 10, 100]
    """
    coco_gt = COCO(gt)
    class_names = []
    for key in coco_gt.cats.keys():
        class_names.append(coco_gt.cats[key]['name'])

    coco_dt = coco_gt.loadRes(predictions)

    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }[iou_type]

    # iou_map = {
    #     0.50: 0,
    #     0.55: 1,
    #     0.60: 2,
    #     0.65: 3,
    #     0.70: 4,
    #     0.75: 5,
    #     0.80: 6,
    #     0.85: 7,
    #     0.90: 8,
    #     0.95: 9
    # }

    area_range_map = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3
    }

    # max_det_map = {
    #     1: 0,
    #     10: 1,
    #     100: -1
    # }


    # COCOEval
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)

    max_det_map = {
        10: 0,
        100: 1,
        1000: -1
    }
    coco_eval.params.maxDets = [10, 100, 1000]
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 5 ** 2], [5 ** 2, 10 ** 2], [10 ** 2, 1e5 ** 2]]
    iou_thres = np.linspace(.05, 0.95, int(np.round((0.95 - .05) / .1)) + 1, endpoint=True)
    coco_eval.params.iouThrs = iou_thres
    iou_map = {}
    for i, iou_t in enumerate(iou_thres):
        iou_t = int(np.round(iou_t*100))/100
        iou_map[iou_t] = i
    print(iou_map)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    if coco_eval is None:
        print("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    # the standard metrics
    results = {
        metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
        for idx, metric in enumerate(metrics)
    }

    if not np.isfinite(sum(results.values())):
        print("Some metrics cannot be computed and is shown as NaN.")

    # Compute per-category AP and AR
    # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
    """
    precision has dims (iou, recall, cls, area_range, max_dets)
    precision.shape == (10, 101, 80, 4, 3)
    recalls has dims (iou, cls, area_range, max_dets)
    recalls.shape == (10, 80, 4, 3)

    Explaination of each dim:
    > dim(iou) = 10, 0.50:0.95
    > dim(recall) = 101, 0:100
    > dim(cls) = 80, len(class_names)
    > dim(area_range) = 4, (all, small, medium, large), area_range index 0: all area ranges
    > dim(max_dets) = 3, (1, 10, 100), max_dets index -1: typically 100 per image
    """
    precisions = coco_eval.eval["precision"]
    recalls = coco_eval.eval["recall"]

    assert len(class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(class_names):
        if iou is None:
            precision = precisions[:, :, idx, area_range_map[area_range], max_det_map[max_det]]
            recall = recalls[:, idx, area_range_map[area_range], max_det_map[max_det]]
        else:
            precision = precisions[iou_map[iou], :, idx, area_range_map[area_range], max_det_map[max_det]]
            recall = recalls[iou_map[iou], idx, area_range_map[area_range], max_det_map[max_det]]

        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")

        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")

        results_per_category.append(("{}".format(name), float(ap * 100), float(ar * 100)))

    results.update({name: (ap, ar) for name, ap, ar in results_per_category})

    # Display results
    max_length_of_class_name = max([len(class_name) for class_name in class_names])
    iStr = ' {class_name:<{max_length_of_class_name}} | precision = {precision:6.3f} | recall = {recall:6.3f}'
    print("")
    for class_name in class_names:
        print(iStr.format(class_name=class_name,
                          max_length_of_class_name=max_length_of_class_name,
                          precision=results[class_name][0],
                          recall=results[class_name][1]))

    return results


if __name__ == "__main__":
    gt = "D:/gitlab/trainingsys/data/ggpccd/COCO/annotation/origin.json"
    predictions = json.load(open("D:/gitlab/trainingsys/zjdet/results/val/228/zjdet_neck_exp_best/exp5/best_predictions.json", "r"))
    det_result_analyse(gt=gt, predictions=predictions, iou=0.55, area_range="all", max_det=1000)
