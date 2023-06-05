# edited by XeatherH
# a function to transform dota to coco and voc , and auto crop the image, and merge or select from coco.
# 2022/1
import shutil
import os
import numpy as np
import cv2
import json
import random
from tqdm import tqdm
from glob import glob

from datetime import datetime

from .data_utils import mkdir, get_pic_id_map, convert_dict2catgories, get_anno_and_sort_imgid

import logging as log

def merge_cocofiles(merge_path, save_dir, mode,new_class=None, categories=None, save_pic=True):
    image_id = 0
    id_c = 0
    info = {'contributor': 'XeatherH',
            'data_created': datetime.now().strftime("%Y.%m.%d"),
            # 'description': 'a merge data of dior and dota.',
            'description': 'a merge data of dota 832.',
            'url': None,
            'version': '1.0',}
    new_images = []
    new_annos = []
    json_save_path = os.path.join(save_dir, 'COCO', 'annotation')
    mkdir(json_save_path)
    if save_pic:
        pic_save_path = os.path.join(save_dir, "images", mode)
        mkdir(pic_save_path)
    if isinstance(merge_path,str):
        merge_path = [ merge_path ]
    # for class_dir in os.listdir(merge_path):
    #     origin_annotation = os.path.join(merge_path, class_dir)
    for origin_annotation in merge_path:

        annotation_file = origin_annotation + f'/COCO/annotation/{mode}.json'

        each_json, image_id, id_c = get_anno_and_sort_imgid(annotation_file, image_id, id_c,new_class=new_class)
        if os.path.exists(origin_annotation + f'/{mode}/object/'):
            origin_pic_dir = origin_annotation + f'/{mode}/object/'
        else:
            origin_pic_dir = origin_annotation + f'/{mode}/'
        for img in each_json['images']:
            new_images.append(img)
            if save_pic:
                origin_pic_path = origin_pic_dir + img['file_name']
                new_pic_path = pic_save_path+'/'+img['file_name']
                shutil.copy2(origin_pic_path, new_pic_path)

        for i in each_json['annotations']:
            new_annos.append(i)
        if not categories:
            categories = []
            for i in each_json['categories']:
                categories.append(i)


    data_dict = {'info': info, "images": new_images, "type": "instances", "annotations": new_annos,
                 "categories": categories}
    with open(json_save_path + f'/{mode}.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    print(f'{mode}.json is saved in {json_save_path}')



def merge(raw_images_dir, annotation_data, categories, save_path, merge_shape, pic_size, merge_num=100):
    """
    merge small picture to big and return coco annotation
    :param raw_images_dir:
    :param annotation_data:
    :param categories:
    :param save_path:
    :param merge_num: all num of picture to merge
    :param merge_shape:  merged shape, eg: 5000
    :param pic_size: small picture size eg: 800
    :return:
        data_dict: annotation info
    """
    new_images = []
    new_annos = []
    bnd_id = 0
    image_id = 0
    for _ in range(merge_num):
        random.shuffle(annotation_data)
        merge_img_info = []
        merge_img = np.zeros((merge_shape, merge_shape, 3))
        edge_pic_need = int((merge_shape-0.1)//pic_size) + 1
        per_cat_need = int((edge_pic_need**2 - 0.1)//len(categories)) + 1
        for cat in categories:
            id = cat['id']
            num = 0
            for image in annotation_data:
                if num == per_cat_need:
                    break
                pic_name = image['file_name']
                img_height = image['height']
                img_width = image['width']
                flag = 0

                file_path = os.path.join(raw_images_dir, pic_name)
                anno_info = image['anno_info']
                for each_box in anno_info:
                    if int(id) == int(each_box[5]):
                        flag = 1
                        break
                if flag == 0 or int(img_height)*int(img_width) == 0:
                    continue
                img = cv2.imread(file_path)
                num += 1
                merge_img_info.append((img, anno_info, pic_name))
        image_id += 1
        random.shuffle(merge_img_info)

        merge_img_info = sorted(merge_img_info, key=lambda x: (x[0].shape[0], x[0].shape[1]), reverse=True)

        y1, ymin = 0, 0
        file_str = ''
        for i, (img_a, anno, pic_name) in enumerate(merge_img_info):
            x_step = i % edge_pic_need
            if i//edge_pic_need == edge_pic_need:
                break
            if x_step == 0:
                x1 = 0
                y1 += ymin
                ymin = img_a.shape[0]
            else:
                ymin = min(ymin, img_a.shape[0])


            y2 = y1 + img_a.shape[0]
            x2 = x1 + img_a.shape[1]
            x2 = merge_shape if x2 > merge_shape else x2
            y2 = merge_shape if y2 > merge_shape else y2

            merge_img[y1:y2, x1:x2] = img_a[0:y2-y1, 0:x2-x1]
            # file_str += pic_name.strip('.jpg')
            file_str += pic_name+'\n'
            for each_anno in anno:
                w = min(each_anno[2], x2 - x1 - each_anno[0])
                h = min(each_anno[3], y2 - y1 - each_anno[1])
                if w < 0.5*each_anno[2] or h < 0.5*each_anno[3]:
                    continue
                box = list([each_anno[0]+x1, each_anno[1]+y1, w, h])
                area = each_anno[4]
                category_id = each_anno[5]

                bnd_id += 1
                new_ann = {'area': area, 'iscrowd': 0, 'image_id': image_id, 'bbox': box,
                           'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                           'segmentation': [[]]}
                new_annos.append(new_ann)
            x1 = x2
        per_file_name = f'merge_{image_id}.jpg'
        file_save_path = save_path + '/' + per_file_name
        cv2.imwrite(file_save_path, merge_img)
        merge_info_path = save_path + '/info/'
        mkdir(merge_info_path)
        with open(merge_info_path+f'/merge_{image_id}.txt', 'w') as f:
            f.write(file_str)
        log.info('img saved : ', file_save_path)

        new_image = {
            'file_name': per_file_name,
            'height': merge_shape,
            'width': merge_shape,
            'id': image_id
        }
        new_images.append(new_image)
    data_dict = {'info': f"{merge_shape}_merge", "images": new_images, "type": "instances", "annotations": new_annos,
                 "categories": categories}
    return data_dict

def cropped_merge_to_big(raw_data, crop_pic_save_path, mode, merge_shape, pic_size, merge_num=100):
    json_save_path = os.path.join(crop_pic_save_path, 'COCO', 'annotation')
    mkdir(json_save_path)
    raw_json_dir = raw_data + f'/COCO/annotation/{mode}.json'
    annotation_data, categories = get_pic_id_map(raw_json_dir)
    raw_images_dir = os.path.join(raw_data, mode)
    save_path = os.path.join(crop_pic_save_path, mode)
    mkdir(save_path)
    if 'val' in mode:
        merge_num = merge_num//3

    json_data = merge(raw_images_dir, annotation_data, categories, save_path, merge_shape, pic_size, merge_num=merge_num)

    with open(json_save_path + f'/{mode}.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    log.info(f'{mode} is saved in {json_save_path}')

def visual_merge(path):
    """
    merge image which cropped to big picture and save
    :param path: cropped picture path
    """
    save_path = os.path.join(path, 'merge_visual')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    merge_array = []
    for i in range(10):
        img_array = np.zeros((5000,5000,3))
        merge_array.append(img_array)

    for file in tqdm(glob(path+'/*.jpg')):
        img = cv2.imread(file)
        pic_name = os.path.split(file)[-1]
        pic_list = pic_name.split('_')   #['yolov3', 'merge','1','x1','y1','x2','y2.jpg'] #yolov3_merge_1_0_0_800_800.jpg
        pic_id = int(pic_list[2])
        x1 = int(pic_list[3])
        y1 = int(pic_list[4])
        x2 = int(pic_list[5])
        y2 = int(pic_list[6].split('.')[0])
        if pic_id <= 10:
            merge_array[pic_id - 1][y1:y2, x1:x2] = img
    for i, img in enumerate(merge_array):
        if not (img == img_array).all():
            file_path = save_path + f'/merge_{i+1}.jpg'
            cv2.imwrite(file_path, img)




if __name__=='__main__':
    mod_l = ['train','val']

    # ################################################################

    # for crop merge and return new pic and json
    merge_num = 3000
    merge_shape = 832
    pic_size = 256
    raw_data = '/home/workspace/data/essential/sarship2'
    crop_pic_save_path = '/home/workspace/data/essential/sarship2_merged832/'
    for mod in mod_l:
        cropped_merge_to_big(raw_data, crop_pic_save_path, mod, merge_shape, pic_size, merge_num=merge_num)

    # 筛选后的图片及annotation，保存的位置
    save_dir = '/home/workspace/data/essential/merged_sarship_832/'
    new_cls_id = {1: 1}
    cate = {'ship': 1}
    merge_list = ['/home/workspace/data/essential/sarship2_merged832/',
                  '/home/workspace/data/essential/sarship_832/']
    for mod in mod_l:
        merge_cocofiles(merge_list, save_dir, mod, new_class=new_cls_id,
                        categories=convert_dict2catgories(cate))

    # ##############################################################################
    # visual pic merged
    path = '/workspace/mmdetection/results/outputs_val'
    visual_merge(path)
