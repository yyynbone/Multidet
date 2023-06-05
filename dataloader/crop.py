# edited by XeatherH
# a function to transform dota to coco and voc , and auto crop the image, and merge or select from coco.
# 2022/1

import os
import numpy as np
import cv2
import json
from glob import glob
from copy import deepcopy
import random
from tqdm import tqdm
from datetime import datetime
from scipy.cluster.vq import kmeans,whiten
from .data_utils import mkdir,Logger, get_pic_id_map, get_box, filter_difficult
from threading import Thread

def filter_array(arr, min, filter_pix,log=None):
    assert arr.ndim == 2, "we only filter array of ndim=2"
    # Filter
    min_num = (arr < min).any(-1).sum()
    if min_num:
        log.info(f'WARNING: Extremely small objects found. {min_num} of {len(arr)} labels are < {min} pixels in size.')
    return arr[(arr > filter_pix).all(-1)]


def kmeans_action(w_h_array_percat, k=1, class_name='all', iter=20, log=None):
    # Kmeans calculation
    log.info(f'Running kmeans for {class_name} on {len(w_h_array_percat)} points...')
    std_ = w_h_array_percat.std(0)  # standard deviation
    # mean_ = w_h_array_percat.mean(0) # sometimes it is not needed
    # std_wh_percat = (w_h_array_percat-mean_)/std_
    std_wh_percat = whiten(w_h_array_percat)
    point, distance = kmeans(std_wh_percat, k, iter=iter)
    # the origin kmeans point
    point *= std_
    log.info(f'now {class_name} kmeans k is {k},  point is {point}, and distortion is {distance}\n')
    assert len(point) == k, f'ERROR: scipy.cluster.vq.kmeans requested {k} points but returned only {len(point)}'

    return point, distance


def loop_kmeans(wh_array_all, distort_thresh, class_name='all',log=None):
    min_point, min_distort = [], 0
    for j in range(1, 20):
        if j > len(wh_array_all) // 2:
            break
        point, distortion = kmeans_action(wh_array_all, k=j, class_name=class_name, iter=50,log=log)
        if j == 1:
            min_distort = distortion
            min_point = point
        else:
            if distortion / min_distort > (1 - distort_thresh):
                break
            else:
                min_distort = distortion
                min_point = point
    return min_point


def kmeans_get_gtbox(annotation_datas, cate, kmeans_filter=5, per_cat_kmeans=True, distort_thresh=0.05,log=None):
    # w_h_list = [[]]*len(cate)  #arr_id of w_h_list  become same  error
    w_h_dict = {}
    for annotation_data in annotation_datas:
        for bbox in annotation_data['anno_info']:
            if bbox[-1] not in w_h_dict.keys():
                w_h_dict[bbox[-1]] = [bbox[2:4]]
            else:
                w_h_dict[bbox[-1]].append(bbox[2:4])
    points = []
    wh_array_all = np.zeros((0, 2))
    for id, w_h_percat in w_h_dict.items():
        class_name = cate[int(id) - 1]['name']
        log.info(f"###########################>>{class_name}<<#############################")
        w_h_array_percat = filter_array(np.array(w_h_percat), kmeans_filter, kmeans_filter,log=log)
        wh_array_all = np.concatenate((wh_array_all, w_h_array_percat), axis=0)
        # wh_array_all = np.vstack((wh_array_all, w_h_array_percat))
        if per_cat_kmeans:
            # point,_ = kmeans_action(w_h_array_percat, class_name=class_name)
            # points.append(point[0])
            point = loop_kmeans(w_h_array_percat, 2 * distort_thresh, class_name=class_name,log=log)
            points.extend(point)
    # all cate kmeans
    min_point = loop_kmeans(wh_array_all, distort_thresh, log=log)

    if per_cat_kmeans:
        points = loop_kmeans(np.array(points), distort_thresh, class_name='k-means of per class', log=log)
        min_point = np.concatenate((min_point, np.array(points)), axis=0)


    log.info('\n\n###########################>>final_point is:<<#############################')
    log.info(min_point)

    return min_point

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleFill=False, scaleup=True, stride=0):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # ##############################################
    # 保持宽高比，eg:(1280,960) to (640,640), 则ratio=0.5,0.5, 先resize成（640，480）
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # ##############################################

    if stride:  # minimum rectangle,
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding，保持pad为32的倍数
    elif scaleFill:  # stretch，不保持宽高比,完全resize
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides，往两边各pad一部分像素
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def label_resize(x, rw=1, rh=1, padw=0, padh=0):
    if isinstance(x,list):
        x = np.array(x)
    x[:, 0] = rw * x[:, 0] + padw # x
    x[:, 1] = rh * x[:, 1] + padh  # y
    x[:, 2] = rw * x[:, 2]  # w
    x[:, 3] = rh * x[:, 3]  # h
    x = x.astype(np.int32)
    return x

def plot_save_img(visual_save_path,visual_img_c,visual_box=[]):
    for visul_b in visual_box:
        start_p = (int(visul_b[0]), int(visul_b[1]))
        end_p = (int(visul_b[0]) + int(visul_b[2]), int(visul_b[1]) + int(visul_b[3]))
        cls_id = str(visul_b[5])
        # try:
        visual_img_c = cv2.rectangle(visual_img_c, start_p, end_p, (0, 0, 255), 2)
        visual_img_c = cv2.putText(visual_img_c, cls_id, start_p, cv2.FONT_HERSHEY_SIMPLEX,
                                   1, (0, 255, 255), 2, cv2.LINE_AA)
        # except:
        #     print(visual_save_path,visual_box)
        #     print(start_p,end_p)
        #     print("now img shape is ",visual_img_c.shape)
    cv2.imwrite(visual_save_path, visual_img_c)

def crop_with_anno(pic_file,
                   save_path,
                   crop_overlap=32,
                   crop_size=800,
                   anno_info=None,
                   kmeans_point=None,
                   thresh=0.6,
                   shake_thresh=0.2,
                   size_thresh=0.9,
                   overlap_thresh=0.25,
                   box_filter=15,
                   resized=True,
                   save_pic=True,
                   drop_edge=False,
                   allowed_hp_rotate=True,
                   bg_in_anno=False,
                   log=None):
    """
    crop picture from big to small
    :param pic_file: big picture file path
    :param save_path: small pic save path
    :param crop_overlap: pixels overlaped
    :param crop_size:
    :param anno_info:
    :param kmeans_point:
    :param thresh:
    :param shake_thresh:
    :return:
    """
    if isinstance(crop_overlap, int):
        crop_overlap = (crop_overlap, crop_overlap)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    pic_name = os.path.split(pic_file)[1]
    img = cv2.imread(pic_file)
    img_height, img_width = img.shape[:2]
    crop_width, crop_height = crop_size[0], crop_size[1]
    crop_p_w, crop_p_h = crop_overlap[0], crop_overlap[1]
    new_images, new_annos = [], []
    image_id, bnd_id = 0, 0
    obj_box_difficult = []
    obj_box = []
    visual_box = []
    rand_w, rand_h = 0, 0
    arr_box = np.zeros((0, 6))

    # add the situation of img and anno flip up and down
    if (img_height-img_width)*(crop_height-crop_width)<0 and allowed_hp_rotate:
        half_pi_rotate = True
        # print('origin:',img.dtype)
        img = np.ascontiguousarray(img.transpose(1,0,2))
        #使用transpose 必须使用ascontiguousarray，否则数据不连续，rectangle报错
        # Layout of the output array img is incompatible with cv::Mat >
        # - Expected Ptr < cv::UMat >for argument 'img'
        # print('after transpose:',img.dtype)
        img_height, img_width = img.shape[:2]
        # pic_name = 'half_pi_rotate_'+pic_name
    else:
        half_pi_rotate = False


    if anno_info:
        anno_info = sorted(anno_info, key=lambda x: (x[0], x[1]))
        arr_box = np.array(anno_info)  # [x1,y1,x2,y2,aera,category]
        if kmeans_point is not None:
            if kmeans_point.any():
                # box_w_mean = arr_box[:, 2].mean()
                # box_h_mean = arr_box[:, 3].mean()
                # p_w = int(box_w_mean / 16)
                # p_w = p_w if p_w > 0 else 1
                # p_w = p_w if p_w < 5 else 5
                # p_h = int(box_h_mean / 16)
                # p_h = p_h if p_h > 0 else 1
                # p_h = p_h if p_h < 5 else 5
                # crop_p_w = p_w * 16
                # crop_p_h = p_h * 16
                # crop_p_w = int(box_w_mean)
                # crop_p_h = int(box_w_mean)
                w_h_mean = arr_box[:, 2:4].mean(0)

                kmeans_point = kmeans_point.astype(np.int32)  # np.version > 1.21 ,astype(int)=np.int64,else int32
                dist = np.linalg.norm(kmeans_point - w_h_mean, axis=1, keepdims=True)

                crop_p_w, crop_p_h = kmeans_point[np.argmin(dist)].tolist()  # if numpy,you get int32 or int64,so to list
                if crop_p_w > crop_width * overlap_thresh:
                    crop_p_w = int(crop_width * overlap_thresh)

                if crop_p_h > crop_height * overlap_thresh:
                    crop_p_h = int(crop_height * overlap_thresh)

                rand_w = int(crop_p_w * shake_thresh)
                rand_h = int(crop_p_h * shake_thresh)
                # log.info(crop_p_h, crop_p_w)

        if half_pi_rotate:
            origin_box = deepcopy(arr_box)
            arr_box[:, :4] = origin_box[:, [1, 0, 3, 2]]
            crop_p_w, crop_p_h = crop_p_h, crop_p_w
            rand_w, rand_h = rand_h, rand_w
    elif bg_in_anno:
        anno_info = True

    w_cr_num = (img_width - crop_p_w) / (crop_width - crop_p_w)
    h_cr_num = (img_height - crop_p_h) / (crop_height - crop_p_h)

    # w_cr_num = (img_width + crop_p_w) / crop_width
    # h_cr_num = (img_height + crop_p_h) / crop_height
    # if w_cr_num > int(w_cr_num) :
    #     w_cr_num = int(w_cr_num)+1
    # if h_cr_num > int(h_cr_num):
    #     h_cr_num = int(h_cr_num) + 1

    crop_box = []

    if not os.path.exists(save_path + "/object") and not bg_in_anno:
        os.makedirs(save_path + "/object")
        os.makedirs(save_path + "/bg")
        # os.makedirs(save_path + "/not_crop")

    if resized:
        crop_thresh = size_thresh
    else:
        crop_thresh = 1
    # if min(w_cr_num, h_cr_num) < crop_thresh:
        # cv2.imwrite(save_path + '/not_crop/' + pic_name, img)

    left, top = 0, 0
    if min(w_cr_num, h_cr_num)< crop_thresh:   # first pad to the crop size
        if  drop_edge or resized:
            return new_images, new_annos if anno_info else crop_box

        else:
            pic_name = 'pad_' + pic_name
            padw, padh = max(crop_width - img_width,0), max(crop_height - img_height,0)  # wh padding
            dh, dw = random.randint(0,padh), random.randint(0,padw)
            top, bottom = int(round(dh - 0.1)), int(round(padh-dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(padw-dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))  # add border
            img_height, img_width = img.shape[:2]
            if anno_info:
                arr_box = label_resize(arr_box, padw=left, padh=top)

    file_path_name_ = pic_name.split('.')
    x_step = 0
    x_flag = 1
    x1 = 0
    while x_flag:
        if x_step > 0:
            if rand_w==0 and x_step==1 and (drop_edge or resized):
                crop_p_w_s = crop_p_w + random.randint(0, crop_width//2)
            else:
                crop_p_w_s = crop_p_w + random.randint(-rand_w, rand_w)
            x_temp = x1
            if x1 < img_width - 2 * crop_width + crop_p_w_s:
                x1 += (crop_width - crop_p_w_s)
            else:
                x_flag = 0
                x1 = img_width - crop_width

            if x1 < 0 or abs(x1-x_temp)<box_filter:
                break
        width_ratio = (img_width - x1)/crop_width
        if width_ratio <= 2 - size_thresh:
            if resized:
                x_flag = 0
            elif drop_edge:
                # if drop out the infomation of edge in order to less cropped pic.
                width_ratio = 1
                x_flag = 0
            else:
                width_ratio = 1
        else:
            width_ratio = 1

        x_step += 1

        y_step = 0
        y_flag = 1
        y1 = 0

        while y_flag:

            if y_step > 0:
                if rand_h==0 and y_step==1 and (drop_edge or resized):
                    crop_p_h_s = crop_p_h + random.randint(0, crop_height//2)
                else:
                    crop_p_h_s = crop_p_h + random.randint(-rand_h, rand_h)
                y_temp = y1
                if y1 < img_height - 2 * crop_height + crop_p_h_s:
                    y1 += (crop_height - crop_p_h_s)
                else:
                    y1 = img_height - crop_height
                    y_flag = 0
                # crop the image
                if y1 < 0 or abs(y1-y_temp)<box_filter:
                    break

            height_ratio = (img_height - y1) / crop_height
            if height_ratio <= 2 - size_thresh:
                if resized:
                    y_flag = 0
                elif drop_edge:
                    # if drop out the infomation of edge in order to less cropped pic.
                    height_ratio = 1
                    y_flag = 0
                else:
                    height_ratio = 1
            else:
                height_ratio = 1

            y_step += 1
            ######################modified to random point as the start point to crop#######################
            '''
            x_step = 0
            x_flag = 2
            x1 = 0
            while x_flag:
                crop_p_w_s = crop_p_w + random.randint(-rand_w, rand_w)
    
                width_ratio = (img_width - x1) / crop_width
                if width_ratio > size_thresh and width_ratio < 2 - size_thresh:
                    x_flag -=1
                else:
                    width_ratio = 1
                    if x_flag == 2:
                        if x_step == 0:
                            pass
                        elif x1 < img_width - 2 * crop_width + crop_p_w:
                            x1 += (crop_width - crop_p_w_s)
                        else:
                            x_flag = 0
                            x1 = img_width - crop_width
                    else:
                        if x1 < img_width - 2 * crop_width + crop_p_w:
                            x1 += (crop_width - crop_p_w_s)
                        else:
                            x_flag = 0
                            x1 = img_width - crop_width
                if x1 < 0:
                    break
                if not resized:
                    width_ratio = 1
    
                x_step += 1
    
                y_step = 0
                y_flag = 2
                y1 = 0
    
                while y_flag:
                    crop_p_h_s = crop_p_h + random.randint(-rand_h, rand_h)
                    height_ratio = (img_height - y1) / crop_height
                    if height_ratio > size_thresh and height_ratio < 2 - size_thresh:
                        y_flag -=1
                    else:
                        height_ratio = 1
                        if y_step == 0:
                            pass
                        elif y1 < img_height - 2 * crop_height + crop_p_h:
                            y1 += (crop_height - crop_p_h_s)
                        else:
                            y1 = img_height - crop_height
                            y_flag = 0
                    # crop the image
                    if y1 < 0:
                        break
                    if not resized:
                        height_ratio = 1
                    y_step += 1
            '''
            ######################modified to random point as the start point to crop###################

            x2 = x1 + int(round(crop_width * width_ratio))
            y2 = y1 + int(round(crop_height * height_ratio))

            save_flag = 1
            check_flag = 1

            per_file_name = f'{file_path_name_[0]}-{x1-left}_{x2-left}_{y1-top}_{y2-top}.{file_path_name_[1]}'
            visual_name = f'{file_path_name_[0]}-{x1-left}_{x2-left}_{y1-top}_{y2-top}_visual.{file_path_name_[1]}'
            # print(per_file_name)

            img_c = img[y1:y2, x1:x2]

            if anno_info and arr_box.shape[0]:
                save_flag = 0
                # only for visual
                visual_box = get_box((x1, y1, x2, y2), arr_box, 0.3)
                # #debug if exist catid
                # if (arr_box[:, 5] == 6).any():
                #     pass
                # else:
                #     return new_images, new_annos if anno_info else crop_box
                if isinstance(thresh, (tuple, list)):
                    obj_box_all = get_box((x1, y1, x2, y2), arr_box, thresh[0])  # small thresh
                    obj_box = get_box((x1, y1, x2, y2), arr_box, thresh[1])
                    obj_box_difficult = np.array([box for box in obj_box_all if box not in obj_box])
                    if len(obj_box_all):
                        save_flag = 1
                    if len(obj_box):
                        check_flag = 0
                else:
                    obj_box_difficult = []
                    obj_box = get_box((x1, y1, x2, y2), arr_box, thresh)
                    if len(obj_box):
                        save_flag = 1
                        check_flag = 0


            if width_ratio !=1 or height_ratio!=1:
                img_c, (rw,rh), (padw, padh) = letterbox(img_c,(crop_height,crop_width))
                per_file_name = 'resized_'+per_file_name
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                if anno_info:
                    if len(obj_box):
                        obj_box = label_resize(obj_box, rw, rh, padw, padh)
                        # check_flag = 1
                    if len(obj_box_difficult):
                        obj_box_difficult = label_resize(obj_box_difficult, rw, rh, padw, padh)
                    if len(visual_box):
                        visual_name = 'resized_' + visual_name
                        visual_box = label_resize(visual_box, rw, rh, padw, padh)

            if save_flag:
                belong_to = 'object'
                crop_box.append([x1, y1, x2, y2])
                if anno_info:
                    image_id += 1
                    new_image = {
                        'file_name': per_file_name,
                        'height': y2 - y1,
                        'width': x2 - x1,
                        'id': image_id
                    }
                    new_images.append(new_image)

                    for each_box in obj_box:
                        bnd_id += 1
                        new_ann = {'area': int(each_box[4]), 'iscrowd': 0, 'image_id': image_id,
                                   'bbox': each_box[:4].tolist(),
                                   'category_id': int(each_box[5]), 'id': bnd_id, 'ignore': 0,
                                   'segmentation': [[]]}
                        # if each_box[2] < box_filter or each_box[3] < box_filter:
                        if each_box[2]*each_box[3] < box_filter**2:
                            new_ann['ignore'] = 1
                        new_annos.append(new_ann)

                    for each_box in obj_box_difficult:
                        bnd_id += 1
                        new_ann = {'area': int(each_box[4]), 'iscrowd': 0, 'image_id': image_id,
                                   'bbox': each_box[:4].tolist(),
                                   'category_id': int(each_box[5]), 'id': bnd_id,
                                   'ignore': 1, 'segmentation': [[]]}
                        if each_box[2] >= 1.5*box_filter and each_box[3] >= 1.5*box_filter:
                            new_ann['ignore'] = 0
                        new_annos.append(new_ann)
            else:
                belong_to = 'bg'
                if bg_in_anno:
                    if anno_info:
                        image_id += 1
                        new_image = {
                            'file_name': per_file_name,
                            'height': y2 - y1,
                            'width': x2 - x1,
                            'id': image_id
                        }
                        new_images.append(new_image)
            if bg_in_anno:
                belong_to = ''

            file_save_path = os.path.join(save_path, belong_to, per_file_name)
            visual_save_path = os.path.join(save_path, belong_to,  visual_name)
            # cv2.imwrite(file_save_path, img_c)
            if len(visual_box) and check_flag:
                visual_img_c = deepcopy(img_c)
                # print(img_c.dtype)
                # print(visual_img_c.dtype)
                # for visul_b in visual_box:
                #     start_p = (int(visul_b[0]), int(visul_b[1]))
                #     end_p = (int(visul_b[0]) + int(visul_b[2]), int(visul_b[1]) + int(visul_b[3]))
                #     cls_id = str(visul_b[5])
                #     visual_img_c = cv2.rectangle(visual_img_c, start_p, end_p, (0, 0, 255), 2)
                #     visual_img_c = cv2.putText(visual_img_c, cls_id, start_p, cv2.FONT_HERSHEY_SIMPLEX,
                #                                1, (0, 255, 255), 2, cv2.LINE_AA)
                # cv2.imwrite(visual_save_path, visual_img_c)
                if save_pic:
                    Thread(target=plot_save_img, args=(visual_save_path, visual_img_c,visual_box)).start()
            # log.info('img saved : ', file_save_path)
            h, w = img_c.shape[:-1]
            if (w, h) == crop_size:
                pass
            else:
                print(file_save_path, h, w)
            if save_pic:
                Thread(target=plot_save_img,args=(file_save_path,img_c)).start()

    return new_images, new_annos if anno_info else crop_box

def crop(raw_images_dir, new_json_info, save_path,
         crop_overlap=32,
         crop_size=800,
         auto_crop=False,
         thresh=0.6,
         kmeans_filter=5,
         shake_thresh=0.2,
         size_thresh=0.9,
         overlap_thresh=0.25,
         resized=True,
         save_pic=True,
         drop_edge=False,
         allowed_hp_rotate=True,
         bg_in_anno=False,
         log=None):
    categories = new_json_info['categories']
    annotation_data = new_json_info['anno_info']
    info = {'contributor': 'XeatherH',
            'data_created': datetime.now().strftime("%Y.%m.%d"),
            'description': f"a crop of {crop_size} with thresh of {thresh} of {new_json_info['info']['description']}",
            'version': '2.0'}

    new_images, new_annos = [], []
    image_id, bnd_id = 0, 0  # at first it has 0 img

    if auto_crop:
        min_point = kmeans_get_gtbox(annotation_data, categories, kmeans_filter=kmeans_filter,log=log)
    else:
        min_point = None
    for image in tqdm(annotation_data):
        pic_name = image['file_name']
        # if pic_name == 'dior_05745.jpg':
        #     pass
        # else:
        #     continue
        anno_info = image['anno_info']
        file_path = os.path.join(raw_images_dir, pic_name)
        images, annos = crop_with_anno(file_path,
                                       save_path,
                                       crop_overlap=crop_overlap,
                                       crop_size=crop_size,
                                       anno_info=anno_info,
                                       kmeans_point=min_point,
                                       thresh=thresh,
                                       shake_thresh=shake_thresh,
                                       size_thresh=size_thresh,
                                       overlap_thresh=overlap_thresh,
                                       resized=resized,
                                       save_pic=save_pic,
                                       box_filter=kmeans_filter,
                                       drop_edge=drop_edge,
                                       allowed_hp_rotate=allowed_hp_rotate,
                                       bg_in_anno=bg_in_anno,
                                       log=log)
        for image_info in images:
            # image id from 1 to m ,
            image_info['id'] += image_id
            new_images.append(image_info)

        for anno_per in annos:
            if anno_per:
                anno_per['image_id'] += image_id
                anno_per['id'] += bnd_id
                new_annos.append(anno_per)

        image_id += len(images)  # now it has m+image_id picture
        bnd_id += len(annos)
    log.info(f'total cropped {image_id} pictures with objects')
    data_dict = {'info': info, "images": new_images, "type": "instances", "annotations": new_annos,
                 "categories": categories}
    return data_dict

def resized_crop_pic_coco(raw_data,
                          crop_pic_save_path,
                          mode_l=['train', 'val'],
                          auto_crop=True,
                          kmeans_filter=5,
                          crop_size=256,
                          crop_overlap=32,
                          iou_thres=0.55,
                          shake_thresh=0.2,
                          size_thresh=0.9,
                          overlap_thresh=0.25,
                          resized=True,
                          save_pic=True,
                          drop_edge=False,
                          allowed_hp_rotate=True,
                          bg_in_anno=False,
                          log=None):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    assert crop_size[0] > 0 and crop_size[1] > 0
    json_save_path = os.path.join(crop_pic_save_path, 'COCO', 'annotation')
    mkdir(json_save_path)

    for mode in mode_l:
        raw_json_dir = raw_data + f'/COCO/annotation/{mode}.json'
        new_json_info = get_pic_id_map(raw_json_dir)
        if os.path.exists(raw_data+'/images'):
            raw_images_dir = os.path.join(raw_data, 'images', mode)
        else:
            raw_images_dir = os.path.join(raw_data, mode)
        save_path = os.path.join(crop_pic_save_path, 'images', mode)
        mkdir(save_path)

        json_data = crop(raw_images_dir,
                         new_json_info,
                         save_path,
                         crop_overlap=crop_overlap,
                         crop_size=crop_size,
                         auto_crop=auto_crop,
                         thresh=iou_thres,
                         kmeans_filter=kmeans_filter,
                         shake_thresh=shake_thresh,
                         size_thresh=size_thresh,
                         overlap_thresh=overlap_thresh,
                         resized=resized,
                         save_pic=save_pic,
                         drop_edge=drop_edge,
                         allowed_hp_rotate=allowed_hp_rotate,
                         bg_in_anno=bg_in_anno,
                         log=log)

        with open(json_save_path + f'/{mode}_difficult.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        log.info(f'{mode}_diffcult is saved in {json_save_path}')
        json_data = filter_difficult(json_data)

        with open(json_save_path + f'/{mode}.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        log.info(f'{mode} is saved in {json_save_path}')





if __name__=='__main__':
    mod_l = ['train','val']

    # for  coco crop
    raw_data = '/home/workspace/data/essential/dota2'
    crop_pic_save_path = f'/home/workspace/data/essential/auto_cropped_3000_dota2'
    # raw_data = '/home/workspace/data/essential/dior'
    # crop_pic_save_path = f'/home/workspace/data/essential/auto_croped_800_dior'
    mkdir(crop_pic_save_path)
    log = Logger(crop_pic_save_path+'/out.log').logger

    resized_crop_pic_coco(raw_data, crop_pic_save_path, mod_l, crop_size=3000, crop_overlap=32,iou_thres=(0.5,0.8), auto_crop=True,kmeans_filter=15, resized=False,log=log)

