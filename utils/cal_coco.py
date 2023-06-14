import numpy as np
import os
import platform
import math
import cv2
import json
from threading import Thread
from copy import deepcopy
from collections import defaultdict
from terminaltables import AsciiTable
from utils.mix_utils import  mkdir
from utils.logger import print_log
from utils.plots import visual_images


def bbox_overlaps(bboxes1, bboxes2, mode='iou', cal_bou=False, eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0] and not cal_bou:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        if cal_bou:
            ious[i, :] = area1[i] / union
        else:
            ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def matched_put_false(iou_and_cat, gt_idx, pred_idx, count_num, pred_p_list, scores=None, flag=0, not_T=True):
    if not_T:
        i, j = 0, 1
    else:
        i, j = 1, 0
    gt_idx_count = np.bincount(gt_idx)
    idx_l = np.where(gt_idx_count == count_num)[0]  # np.where返回的是元组（x,)
    for idx in idx_l:
        # modified of not select the first index, but select the one with best score
        if scores is None:
            pred_p = pred_idx[gt_idx == idx][0]
        else:
            pred_p_l = pred_idx[gt_idx == idx]
            score = scores[idx, pred_p_l] if not_T else scores[ pred_p_l, idx]
            pred_p = pred_p_l[np.argsort(score)[-1]]
        if pred_p in pred_p_list[j]:
            continue

        pred_p_list[i].append(idx)
        pred_p_list[j].append(pred_p)
        # iou_and_cat = np.delete(iou_and_cat, idx,0)
        # iou_and_cat = np.delete(iou_and_cat, pred_p, 1)
        if not_T:
            iou_and_cat[idx, :] = False
            iou_and_cat[:, pred_p] = False
        else:
            iou_and_cat[:, idx] = False
            iou_and_cat[pred_p, :] = False
        flag = 1
        if iou_and_cat.any() == False:
            return iou_and_cat, pred_p_list,2
    return iou_and_cat, pred_p_list, flag

def all_rm_points_array(iou_and_cat,pred_p_list,scores=None):
    """
    modified the function of rm_points_array,with less recursion depth
    :param iou_and_cat:  r*w
    :param pred_p_list: [[],[]]
    :param scores: r*w
    :return:
    """
    if iou_and_cat.any() == False:
        return iou_and_cat, pred_p_list
    for count_num in range(1,iou_and_cat.shape[0]+1):
        flag = 0
        gt_idx, pred_idx = np.where(iou_and_cat == True)
        iou_and_cat, pred_p_list, flag = matched_put_false(iou_and_cat, gt_idx, pred_idx, count_num, pred_p_list, scores, flag)
        if flag == 2:
            return iou_and_cat, pred_p_list
        elif flag == 1:
            gt_idx, pred_idx = np.where(iou_and_cat == True)

        iou_and_cat, pred_p_list, flag = matched_put_false(iou_and_cat, pred_idx, gt_idx, count_num, pred_p_list, scores, flag, False)
        if flag==2:
            return iou_and_cat, pred_p_list

        if flag==1:
            iou_and_cat, pred_p_list = all_rm_points_array(iou_and_cat,pred_p_list,scores)
            if iou_and_cat.any() == False:
                return iou_and_cat, pred_p_list

# without pred box of reused
def cal_recall(gt, bbox_result, iouthresh=0, cal_bou=False, map_match=False):
    """
    calculate gt of recalled and not recalled
    :param gt:  array shape (n,5)  box and category
    :param bbox_result:  array shape (k,6) box,score and category
    :param iouthresh:
    :return: is_recalled (array)  shape (n,2)   cate_id and is_recalled or not ,
         1 means recalled and 0 means not recalled
    """
    is_recalled = np.zeros((gt.shape[0], 2))
    all_recalled = np.zeros(gt.shape[0])
    gt_box = gt[:, :4]
    bbox = bbox_result[:, :4]
    iou_array = bbox_overlaps(gt_box, bbox,cal_bou=cal_bou)   # (n,k)
    cat_id = gt[:,-1:]   # shape of (n,1)
    gt_num = gt_box.shape[0]

    iou_and_cat = np.logical_and(iou_array >= iouthresh, bbox_result[:, -1][None].repeat(gt_num,axis=0) == cat_id)
    if map_match:
        pred_p_list = [[] for _ in range(2)]
        iou_and_cat, pred_p_list = all_rm_points_array(iou_and_cat,pred_p_list,iou_array)
        for i in range(gt_num):
            cat_id = gt[i, -1]
            is_recalled[i, 0] = int(cat_id)
            if i in pred_p_list[0]:
                is_recalled[i, 1] = 1

        iou_and_cat = np.logical_and(iou_array >= iouthresh, bbox_result[:, -1][None].repeat(gt_num,axis=0)>=0)
        pred_p_list = [[] for _ in range(2)]
        iou_and_cat, pred_p_list = all_rm_points_array(iou_and_cat, pred_p_list, iou_array)
        for i in range(gt_num):
            if i in pred_p_list[0]:
                all_recalled[i] = 1
    else:
        is_recalled[:, 0] = gt[:, -1]
        is_recalled[:, 1] = np.logical_or.reduce(iou_and_cat, axis=1).astype(np.int8)
        all_recalled = np.logical_or.reduce(iou_array >= iouthresh, axis=1).astype(np.int8)

    return is_recalled, all_recalled

def box2points(box, dim1):
    box = np.array(box).reshape((-1, dim1))
    assert box.ndim == 2, f' bboxes ndim should be 2, but box is {box}.'
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def bbox_select(bbox_result, score_thr=(0.01, 0.1),maxDet=100):
    bbox_result = bbox_result[np.argsort(bbox_result[:, 4])[::-1][:maxDet]]
    # auto_score_weight = {1: 1, 2: 3, 3: 19, 4: 44, 5: 563}   # if max_num weight =1 ,other_num weight = (max_num/other_num)**1.5
    # auto_score_weight = {1: 1, 2: 1, 3: 2**2.7, 4: 2**3.5, 5: 2**17.1}   # if max_num weight =1 ,other_num weight = 2**[(mean_num/other_num)-1] ,mean is 471.8

    """
    # if max_num weight =1 ,
    # other_num weight = clsnum**[(max_num/other_num)/clsnum], 1439 661 129 104 26
    cat_id = [1, 2, 3, 4, 5]
    obj_num = [5868, 2858, 842, 549, 58]  # number  5868   2858    842    549   58 from train
    area_weight = [1, 1, 1, 6, 3]
    func = lambda x, y: math.exp((max(obj_num) / y / x - 1) / 10)
    weight_list = list(map(func, obj_num, area_weight))
    auto_score_weight = dict(zip(cat_id, weight_list))
    # print(auto_score_weight)

    # box_result_weight = deepcopy(bbox_result)
    # box_result_weight[:, 4] *= [auto_score_weight[i] for i in box_result_weight[:, 5]]  #  now box_result_weight is just list, not array
    # new_score = bbox_result[:, 4] * [auto_score_weight[i] for i in bbox_result[:, 5]]
    # bbox_result = bbox_result[np.where(np.array(new_score) > score_thr)]
    bbox_result[:, 4] *= [auto_score_weight[i] for i in bbox_result[:, 5]]
    """

    # score_flag = 0
    # select_index = np.ones(bbox_result.shape[0], dtype=int)
    # for i, (score, cat) in enumerate(bbox_result[:, 4:6]):
    #     if cat in [1, 2]:
    #         if score > score_thr*2:  #
    #             score_flag = 1
    #         elif score < 0.05:
    #             select_index[i] = 0    # pop score is less than 0.05 of cat 1 and 2, 3
    #
    # bbox_result = bbox_result[select_index == 1]
    #
    # if not score_flag:
    #     score_thr = min((np.mean(bbox_result[:, 4]) + np.median(bbox_result[:, 4])) / 2, score_thr)



    # if bbox_result.any():
    if isinstance(score_thr, tuple):
        # select_index = np.ones(bbox_result.shape[0], dtype=int)
        # for i, (score, cat) in enumerate(bbox_result[:, 4:6]):
        #     if cat in [1, 2]:
        #         if score < score_thr:
        #             select_index[i] = 0  # pop score is less than 0.05 of cat 1 and 2, 3
        #     if score <= 0.01:
        #         select_index[i] = 0
        bbox_result = bbox_result[bbox_result[:, 4]>=score_thr[0]]
        # count =  np.bincount(bbox_result[:, 5].astype(np.int64))
        # idx = list(range(1,len(count)+1))[count>5*gt_num]
        # for cat_id in idx:
        #     bbox_result = bbox_result[bbox_result[:, 4]>=0.1 & bbox_result[:, 5]==cat_id]
        if bbox_result.size:
            score_mean = np.mean(bbox_result[:, 4])
            score_thr = min(max(score_mean, np.median(bbox_result[:, 4])), score_thr[1])
        else:
            score_thr = score_thr[0]

    # score_thr = np.median(bbox_result[..., 4])

    bbox_result = bbox_result[np.where(bbox_result[:, 4] >= score_thr)]
    return bbox_result

def filter_same_image(image_file, image_file_names):
    if image_file in image_file_names:
        return True
    return False

def load_annotations(coco):
    """Load annotation from COCO style annotation file.

    Args:
        coco (class COCO): COCO(annotation file).

    Returns:
        data_infos [list[dict]]: Annotation info from COCO api.
    """

    cat_ids, class_name = [], []
    for k,v in coco.cats.items():
        cat_ids.append(v['id'])
        class_name.append(v['name'])
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    img_ids = coco.getImgIds()
    data_infos = []
    image_file_names = []
    total_ann_ids = []
    # print("in coco load_annotation, img number is: ", len(self.img_ids))
    repeat_image_file = []
    for i in img_ids:
        info = coco.loadImgs([i])[0]
        if platform.system()=='Windows':
            info['file_name'] = info['file_name'].replace(':',"_")
        info['filename'] = info['file_name']
        if filter_same_image(info['file_name'], image_file_names):
            repeat_image_file.append(info['file_name'])
            continue
        image_file_names.append(info['file_name'])
        data_infos.append(info)
        ann_ids = coco.getAnnIds(imgIds=[i])
        total_ann_ids.extend(ann_ids)
    assert len(set(total_ann_ids)) == len(
        total_ann_ids), f"Annotation ids  are not unique!"
    print(f'total image file {len(img_ids)},repeat image file {len(repeat_image_file)},'
              f'now image file {len(data_infos)}')
    return data_infos, cat_ids, cat2label, class_name

def select_score_from_json(pred_json,score_thresh=0.1):
    new_annos = []
    with open(pred_json, 'r') as f:
        annos = json.load(f)
    for anno in annos:
        if anno['score']<score_thresh:
            # annos.remove(anno)  # 很慢
            continue
        else:
            new_annos.append(anno)
    return new_annos

def visual_return(cocoeval, anno, save_dir, img_prefix, class_area=None, iou_id=0, score_thresh=0.001, cal_bou=False, map_match=False, logger=None, save_visual=False):
    # class_area=[[1,4,7],[3,5,8,9],[2,6]] means small, medium, big category

    data_infos, cat_ids, cat2label, classnames = load_annotations(anno)
    if save_visual:
        visual_save_path = os.path.join(save_dir, f'coco_visual_images_{score_thresh}'.replace(".", "_"))
        mkdir(visual_save_path)

        for is_igmatch in ['matched', 'false_pred', 'not_matched']:
            save_path = os.path.join(visual_save_path, is_igmatch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    show_dict = defaultdict(list)
    aRng = cocoeval.params.areaRng
    # aRng = [15 ** 2, 1e5 ** 2]
    cat_recalled = []
    all_cat_pred = []
    all_correct_recall = []

    for e in cocoeval.evalImgs:
        if not e is None:
            if not class_area:
                areaRs = aRng[:1]
            else:
                for i in range(3):
                    if e['category_id'] in class_area[i]:
                        areaRs = aRng[i+1:]
            for areaR in areaRs:
                if e['aRng'] == areaR:
                    dtScores = np.array(e['dtScores'])
                    dtm = e['dtMatches'][iou_id]
                    gtm = e['gtMatches'][iou_id]
                    dtIg = e['dtIgnore'][iou_id]
                    gtIg = e['gtIgnore']
                    det_id = np.array(e['dtIds'])
                    groud_id = np.array(e['gtIds'])
                    det_select = np.logical_and(det_id, np.logical_not(dtIg))
                    # det_select = np.logical_and(np.logical_and(det_id, np.logical_not(dtIg)), dtScores > auto_score_thresh[e['category_id']])
                    dtScores_select = dtScores[det_select]
                    ds_id = det_id[det_select]
                    gt_select = np.logical_and(groud_id, np.logical_not(gtIg))
                    gs_id = groud_id[gt_select]
                    gtm = gtm[gt_select]
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    true_det = det_id[tps]
                    false_det = det_id[fps]
                    show_dict[e['image_id']].append([e['category_id'], gs_id, ds_id, dtScores_select, true_det, false_det, gtm])
    print_log('coco eval checked and collated \n', logger=logger)
    for imageid, show_info in show_dict.items():

        gt_result = []
        bbox_result = []
        is_igmatch = 'matched'

        for info in show_info:
            cat_id, g_id, d_id, dtscores, true_det, false_det, gtm = info
            if 0 in gtm: # gtm is a list of the id of pred which matched the gt,if not matched, id is 0
                is_igmatch = 'not_matched'
            elif false_det.shape[0]:
                is_igmatch = 'false_pred'

            gts = cocoeval._gts[imageid, cat_id]
            dts = cocoeval._dts[imageid, cat_id]
            for gid, gt in zip(g_id, gtm):
                for i in gts:
                    if gid == i['id']:

                        # i['bbox'].append(cat_id)   # here, the gts was modified, so  error unable to get repr for <class 'mmdet,datasets.Remote_coco.RemoteDataset'>

                        box_gt = deepcopy(i['bbox'])
                        box_gt.append(cat_id)

                        if gt:
                            box_gt.append(1)
                        else:
                            box_gt.append(0)
                        gt_result.append(box_gt)
                        break
            for did in d_id:
                for i in dts:
                    if did == i['id']:
                        box_sc = []
                        for j in i['bbox']:
                            j = max(int(j), 0)
                            box_sc.append(j)
                        box_sc.extend([i['score'], cat_id])

                        if did in true_det:
                            box_sc.append(1)
                        else:
                            box_sc.append(0)

                        bbox_result.append(box_sc)

        gt_result = box2points(gt_result, 6)
        bbox_result = box2points(bbox_result, 7)
        # bbox_result = bbox_select(bbox_result, score_thresh)  # [[306.      278.      433.      425.        0.83548   5  ], ...]
        # print(gt_result, bbox_result)
        iou_thres = cocoeval.params.iouThrs[iou_id]
        # box_scale = 1 / iou_thres
        # iou_thres =1/( (1+math.sqrt(box_scale))/2 )**2 if cal_bou else iou_thres
        iou_thres  = 4 * iou_thres / (1 + math.sqrt(iou_thres))**2 if cal_bou else iou_thres

        cat_corrected, all_corrected = cal_recall(gt_result[:, :-1], bbox_result[:, :-1],
                                                  iouthresh=iou_thres, cal_bou=cal_bou,
                                                  map_match=map_match)

        cat_recalled.append(cat_corrected)
        all_cat_pred.append(bbox_result[:, :-1])
        all_correct_recall.append(all_corrected)

        if save_visual:

            img_name = data_infos[imageid - 1]['file_name']
            img_path = os.path.join(img_prefix, img_name)

            all_gt_num = len(cat_corrected)
            matched_num =  cat_corrected[:, 1].sum()
            if not cat_corrected[:, 1].all():
                is_igmatch = 'not_matched'
                print(f'{img_name} loss match')
            if all_gt_num != matched_num:
                print(f'{all_gt_num-matched_num} loss match in {img_name}')
                # print(cat_corrected)

            save_path = os.path.join(visual_save_path, is_igmatch)
            outfile = os.path.join(save_path, img_name)
            im = cv2.imread(img_path)  # BGR
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            visual_gt = deepcopy(gt_result)
            visual_pred = deepcopy(bbox_result)
            visual_gt[:, -2] -= 1 # category from 1, so substract 1
            visual_pred[:, -2] -= 1 # category from 1
            # print(f'save {outfile}')
            Thread(target=visual_images, args=(im, visual_pred, visual_gt, img_path, outfile, classnames),
                   daemon=False).start()  # daemon为真，则表示守护线程，即程序不会因为子进程没跑完而堵塞，直接主进程停止
            # visual_images(im, gt_result, bbox_result, img_path, outfile, classnames)

    cat_recalled = np.concatenate(cat_recalled,axis=0)
    all_cat_pred = np.concatenate(all_cat_pred, axis=0)
    all_correct_recall = np.concatenate(all_correct_recall, axis=0)

    row_header = ['category', 'gt_num', 'precision_num', 'recall_num', 'Recall', 'Precision']
    table_data = [row_header]
    # for cat in np.unique(cat_recalled[:,0]):
    # for cat in self.show_id_list:
    for cat in cat_ids:
        gt_count = np.sum(cat_recalled[:, 0] == cat)
        pred_count = np.sum(all_cat_pred[:, -1] == cat)
        recall_count = np.sum( np.logical_and(cat_recalled[:, 0] == cat, cat_recalled[:, 1]) )
        recall_per_cat = recall_count/max(gt_count,1e-20)
        precision_per_cat = recall_count / max(pred_count, 1e-20)
        # classname = classnames[sorted(self.cat_ids).index(cat)] if classnames else cat
        classname = classnames[cat2label[cat]] if classnames else cat
        row = [f'{classname}', f'{gt_count}', f'{pred_count}', f'{recall_count}',
               f'{float(recall_per_cat):0.3f}', f'{float(precision_per_cat):0.3f}']

        table_data.append(row)
    recall = all_correct_recall.sum() / cat_recalled.shape[0]
    precision = all_correct_recall.sum() / all_cat_pred.shape[0]
    all_row = ['all', cat_recalled.shape[0], all_cat_pred.shape[0], int(all_correct_recall.sum()), f'{float(recall):0.3f}',
               f'{float(precision):0.3f}']
    table_data.append(all_row)

    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)