# edited by XeatherH
# a function to transform dota to coco and voc , and auto crop the image, and merge or select from coco.
# 2022/1
import shutil
import os
import numpy as np
import cv2
import json
import math
from copy import deepcopy
import random
from tqdm import tqdm
from glob import glob
from threading import Thread
from datetime import datetime
from pathlib import Path
import logging as log
import logging.handlers
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
from xml.dom.minidom import Document
import copy
import platform

def check_platform():
    plat_sys = ['Windows', 'Linux', 'Darwin']
    assert platform.system() in plat_sys, f'{platform.system()}is not support and recognized'
    return platform.system()

class Logger(object):
    level_relations = {
        'debug':log.DEBUG,
        'info':log.INFO,
        'warning':log.WARNING,
        'error':log.ERROR,
        'crit':log.CRITICAL
    }#日志级别关系映射
    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = log.getLogger(filename)
        format_str = log.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = log.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = log.handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

##yolotxt2voc
def save_to_xml(pic_name, save_path, size_info, objects_axis, pic_dir=''):
    im_depth = size_info['depth']
    im_width = size_info['width']
    im_height = size_info['height']
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode(f'{pic_dir}')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(f'{pic_name}')
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('dota2.0'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('XeatherH'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(objects_axis[i][-2]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('2point'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode(objects_axis[i][-1]))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('xmin')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('ymin')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('xmax')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('ymax')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

        # x2 = doc.createElement('x2')
        # x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
        # bndbox.appendChild(x2)
        # y2 = doc.createElement('y2')
        # y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
        # bndbox.appendChild(y2)
        #
        # x3 = doc.createElement('x3')
        # x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
        # bndbox.appendChild(x3)
        # y3 = doc.createElement('y3')
        # y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
        # bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent='\t'))
    f.close()


def GetPoly4FromPoly5(poly):
    distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i
                 in range(int(len(poly) / 2 - 1))]
    distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
    pos = np.array(distances).argsort()[0]
    count = 0
    outpoly = []
    while count < 5:
        # print('count:', count)
        if (count == pos):
            outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
            outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
            count = count + 1
        elif (count == (pos + 1) % 5):
            count = count + 1
            continue

        else:
            outpoly.append(poly[count * 2])
            outpoly.append(poly[count * 2 + 1])
            count = count + 1
    return outpoly

def calchalf_iou(poly1, poly2):
    """
        It is not the iou on usual, the iou is the value of intersection over poly1
    """
    inter_poly = poly1.intersection(poly2)
    inter_area = inter_poly.area
    poly1_area = poly1.area
    half_iou = inter_area / poly1_area
    return inter_poly, half_iou

def get_box_from_4p(x1, y1, x2, y2, box, thresh=0.5):
    """
    4 points img(sub img) ((x1, y1), (x2, y1), (x2, y2), (x1, y2)) and box(absolute location) , iou calculate and return the box relative location in the img
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param box:
    :param thresh:
    :return:
    """
    try:
        import shapely.geometry as shgeo
    except:
        return 0
    sub_obj_b_l = []
    imgpoly = shgeo.Polygon([(x1, y1), (x2, y1), (x2, y2),
                             (x1, y2)])
    for obj_b in box:
        poly = [(obj_b[2 * i], obj_b[2 * i + 1]) for i in range(4)]
        gtpoly = shgeo.Polygon(poly)
        if gtpoly.area <= 0:
            continue
        inter_poly, half_iou = calchalf_iou(gtpoly, imgpoly)

        if half_iou == 1:
            sub_obj_b = obj_b
        elif half_iou <= 0:
            continue
        else:
            inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
            out_poly = list(inter_poly.exterior.coords)[0: -1]
            if len(out_poly) < 4:
                continue
            out_poly2 = []
            for i in range(len(out_poly)):
                out_poly2.append(out_poly[i][0])
                out_poly2.append(out_poly[i][1])

            if len(out_poly) == 5:
                # print('==========================')
                out_poly2 = GetPoly4FromPoly5(out_poly2)
            elif len(out_poly) > 5:
                """
                    if the cut instance is a polygon with points more than 5, we do not handle it currently
                """
                continue
            sub_obj_b = choose_best_pointorder_fit_another(out_poly2, obj_b[:8])
            sub_obj_b = list(sub_obj_b) + obj_b[-2:]


        for i in range(4):
            # print(sub_obj_b)
            if sub_obj_b[2*i] < x1:
                sub_obj_b[2 * i] = 0
            elif sub_obj_b[2*i] > x2:
                sub_obj_b[2 * i] = x2 - x1
            else:
                sub_obj_b[2 * i] -= x1

            if sub_obj_b[2 * i + 1] < y1:
                sub_obj_b[2 * i + 1] = 0
            elif sub_obj_b[2 * i + 1] > y2:
                sub_obj_b[2 * i + 1] = y2 - y1
            else:
                sub_obj_b[2 * i + 1] -= y1

        if half_iou < thresh:
            sub_obj_b[-1] = '2'
        sub_obj_b_l.append(sub_obj_b)
    return sub_obj_b_l

def clip_image(self, file_idx, image, boxes_all, width, height, stride_w, stride_h, save_dir):
    min_length = 1e10
    max_length = 1
    if len(boxes_all) > 0:
        shape = image.shape
        for start_h in range(0, shape[0], stride_h):
            for start_w in range(0, shape[1], stride_w):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width
                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                box[:, 4] = boxes[:, 4] - top_left_col
                box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                box[:, 5] = boxes[:, 5] - top_left_row
                box[:, 7] = boxes[:, 7] - top_left_row
                box[:, 8] = boxes[:, 8]
                center_y = 0.25 * (box[:, 1] + box[:, 3] + box[:, 5] + box[:, 7])
                center_x = 0.25 * (box[:, 0] + box[:, 2] + box[:, 4] + box[:, 6])

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)
                if len(idx) > 0 and (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                    mkdir(os.path.join(save_dir, 'images'))
                    img = os.path.join(save_dir, 'images',
                                       "%s_%04d_%04d.png" % (file_idx, top_left_row, top_left_col))
                    cv2.imwrite(img, subImage)

                    mkdir(os.path.join(save_dir, 'labeltxt'))
                    xml = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    pic_path = os.path.split(img)
                    self.save_to_xml(pic_path[0], pic_path[1], xml,subImage.shape[0], subImage.shape[1], box[idx, :])

def mkdir(path,rm=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if rm:
            shutil.rmtree(path)
            os.makedirs(path)

def bbox_overlaps(bboxes1, bboxes2):
    """Calculate the overlap between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
    Returns:
        overlap(ndarray): shape (n, 1)
    """

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    overlap = 0
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1

    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
    return overlap

def get_box(cropped_point, origin_box,iou_thres):
    """
    get new box axis of cropped picture
    :param cropped_point->tuple: the cropped point of new picture in old big picture axis, (left,up,right,down)
    :param origin_box:
    :param iou_thres:
    :return:
    """
    x1, y1, x2, y2 = cropped_point
    box = deepcopy(origin_box)   # 防止list or array赋值，只copy地址，修改后两者值同步被修改的问题
    box = box.astype(np.int32)
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]

    area = bbox_overlaps(box[:, :4], np.array([[x1, y1, x2, y2]]))
    # idx = np.where(np.logical_or(area >= iou_thres * box[:, 4], area >= area_min))
    idx = np.where(area >= iou_thres * box[:, 4])
    box = box[idx]
    if box.any():
        box[:, 4] = area[idx]
        box[:, 0] = np.where(box[:, 0] > x1, box[:, 0], x1)
        box[:, 1] = np.where(box[:, 1] > y1, box[:, 1], y1)
        box[:, 2] = np.where(box[:, 2] < x2, box[:, 2], x2) - box[:, 0]
        box[:, 3] = np.where(box[:, 3] < y2, box[:, 3], y2) - box[:, 1]
        box[:, 0] = box[:, 0] - x1
        box[:, 1] = box[:, 1] - y1
    # judge if the new box of w,h is more the half of the origin
    if box.any():
        box = box[np.logical_and.reduce(box[:,2:4] >0.5*origin_box[idx][:,2:4], axis=1)]
    return box

def get_pic_id_map(annotation_file):
    """
    a function which get infomation from coco json file
    :param annotation_file: coco file
    :return:
        annotation_data:[{'file_name',   ;
            'height',   ;
            'width',   ;
            'anno_info', [[xmin, ymin, o_width, o_height, area, category_id],...]},
            ...]
        categories:all categories in the coco file
    """
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    new_jsoninfo = {}
    new_jsoninfo['info'] = dataset['info']
    new_jsoninfo['categories'] = dataset['categories']
    annotation_data = []
    anno_box_data = {}
    for key in range(len(dataset['images'])):
        anno_box_data[key+1] = []
    for anno in dataset['annotations']:
        key = anno['image_id']
        anno_info = anno['bbox']
        anno_info.append(int(anno['area']))
        anno_info.append(int(anno['category_id']))
        anno_box_data[key].append(anno_info)
    for image in dataset['images']:
        image_anno = {}   # 在循环里申明，重分配内存，防止值被覆盖
        image_anno['file_name'] = image['file_name']
        image_anno['height'] = image['height']
        image_anno['width'] = image['width']
        image_anno['anno_info'] = anno_box_data[image['id']]
        annotation_data.append(image_anno)
    new_jsoninfo['anno_info'] = annotation_data
    return new_jsoninfo

def get_anno_and_sort_imgid(annotation_file, image_id, id_c,new_class=None, select_cat=None):
    """
    get info from annotation_file, and sort the image id
    :param annotation_file: coco file which want to get information
    :param image_id: the first picture id which selected in the cocofile
    :param id_c: the first id of object in the picture
    :return:
        dataset: annotation information ->dict
        image_id: the last picture id
        id_c: the last object id
    """
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)

    if select_cat is not None:
        dataset = filter_cate(dataset, select_cat)

    annotation = []
    images = []
    img_id_map = {}
    for ann in dataset['annotations']:
        if new_class:
            if ann['category_id'] in new_class:
                ann['category_id'] = int( new_class[ ann['category_id'] ] )
            else:
                continue
        else:
            ann['category_id'] = int(ann['category_id'])
        id_c += 1
        ann['id'] = id_c
        if ann['image_id'] not in img_id_map.keys():
            image_id += 1
            img_id_map[ann['image_id']] = image_id
        ann['image_id'] = img_id_map[ann['image_id']]
        annotation.append(ann)
    for image in dataset['images']:
        if image['id'] in img_id_map.keys():
            image['id'] = img_id_map[image['id']]
            images.append(image)
    dataset['annotations'] = annotation
    dataset['images'] = images

    return dataset, image_id, id_c

def merge_cocofiles(merge_path, save_dir, mode,new_class=None, categories=None, save_pic=True, pic_suffix=None, select_cat=None, area_filter=400):
    image_id = 0
    id_c = 0
    info = {'contributor': 'XeatherH',
            'data_created': datetime.now().strftime("%Y.%m.%d"),
            # 'description': 'a merge data of dior and dota.',
            'description': 'a merge data of',
            'url': None,
            'version': '2.0',}
    new_images = []
    new_annos = []
    json_save_path = os.path.join(save_dir, 'COCO', 'annotation')
    mkdir(json_save_path)

    if isinstance(merge_path,str):
        merge_path = [ merge_path ]
        new_class = [new_class]
        select_cat = [select_cat]
        pic_suffix = [pic_suffix]
    else:
        lenth = len(merge_path)
        # for attr in ['new_class', 'select_cat', 'pic_suffix']:
        #     if eval(attr) is None:
        #
        #         # exec(f'{attr}={[None]*lenth}')  # exec在函数内部修改无效，
        #         # 需要用到其它参数，exec(source, globals, locals),globals只能实现全局变量的修改，
        #         # locals 也无法实现函数内局部变量修改（即只能重新赋值变量名不能重复）
        #         exec(f'{attr}={[None] * lenth}', globals())
        if new_class is None:
            new_class = [new_class]*lenth
        if select_cat is None:
            select_cat = [select_cat]*lenth
        if pic_suffix is None:
            pic_suffix = [pic_suffix]*lenth

    pic_save_path = os.path.join(save_dir, "images", mode)
    mkdir(pic_save_path)

    content = ''
    # for root_dir in merge_path:
    #     with open(os.path.join(root_dir, 'images', f'{mode}.txt'), 'r') as f:
    #         content += f.read()

    # for class_dir in os.listdir(merge_path):
    #     origin_annotation = os.path.join(merge_path, class_dir)
    for i,origin_annotation in enumerate(merge_path):
        annotation_file = os.path.join(origin_annotation, f'COCO/annotation/{mode}.json')
        # print(origin_annotation,annotation_file)
        if os.path.exists(annotation_file):
            each_json, image_id, id_c = get_anno_and_sort_imgid(annotation_file, image_id, id_c,new_class=new_class[i],select_cat=select_cat[i])
            if os.path.exists(os.path.join(origin_annotation,'images', f'{mode}/object/')):
                origin_pic_dir = os.path.join(origin_annotation, 'images', f'{mode}/object/')
            else:
                origin_pic_dir = os.path.join(origin_annotation, 'images', f'{mode}/')
            for img in each_json['images']:
                origin_pic_path = origin_pic_dir + img['file_name']
                if pic_suffix[i] is not None:
                    img['file_name'] = pic_suffix[i] + img['file_name']
                new_images.append(img)
                if save_pic:
                    new_pic_path = pic_save_path+'/'+img['file_name']
                    content += new_pic_path
                    shutil.copy2(origin_pic_path, new_pic_path)
                    # Thread(target=shutil.copy2, args=(origin_pic_path, new_pic_path)).start()
                else:
                    content += origin_pic_path

            for an in each_json['annotations']:
                new_annos.append(an)
            if not categories:
                categories = []
                for cat in each_json['categories']:
                    categories.append(cat)
            info['description'] += each_json['info']['description']


    data_dict = {'info': info, "images": new_images, "type": "instances", "annotations": new_annos,
                 "categories": categories}
    with open(json_save_path + f'/{mode}.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    print(f'{mode}.json is saved in {json_save_path}')

    # with open(os.path.join(save_dir, 'images', f'{mode}.txt'), 'w') as f:
    #     f.write(content+'\n')
    cat_count = calcu_json_cat_count(json_save_path + f'/{mode}.json', area_filter=area_filter)
    print(f'json cat count is:[all, ignore, under filter {area_filter}, upper filter {area_filter}]\n')
    print(cat_count)

def filter_difficult(dataset):
    bnd_id = 1
    new_anno = []
    new_image = []
    new_image_id = []
    for ann in dataset['annotations']:
        if ann['ignore']==0:
            ann['id'] = bnd_id
            ann_image_id = ann['image_id']
            if ann_image_id not in new_image_id:
                new_image_id.append(ann_image_id)
            ann['image_id'] = int(new_image_id.index(ann_image_id)) + 1
            new_anno.append(ann)
            bnd_id += 1

    for image_info in dataset['images']:
        if image_info['id'] in new_image_id:
            image_info['id'] = int(new_image_id.index(image_info['id'])) + 1
            new_image.append(image_info)
    dataset['annotations'] = new_anno
    dataset['images'] = new_image
    return dataset

def filter_cate(dataset,new_cat):
    bnd_id = 1
    new_anno = []
    new_image = []
    new_image_id = []
    for ann in dataset['annotations']:
        if ann['category_id'] in new_cat:
            ann_image_id = ann['image_id']
            if ann_image_id not in new_image_id:
                new_image_id.append(ann_image_id)

    for image_info in dataset['images']:
        if image_info['id'] in new_image_id:
            image_info['id'] = int(new_image_id.index(image_info['id'])) + 1
            new_image.append(image_info)

    for ann in dataset['annotations']:
        if ann['image_id'] in new_image_id:
            ann['id'] = bnd_id
            ann['image_id'] = int(new_image_id.index(ann['image_id'])) + 1
            new_anno.append(ann)
            bnd_id += 1
    dataset['annotations'] = new_anno
    dataset['images'] = new_image
    return dataset

def calcu_json_cat_count(annotation_file, area_filter=100):
    """
    select annotation info of new category from coco file
    :param annotation_file:  origin coco file
    :return:
        dataset: annotation info
    """
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    class_count = {}

    for cat in dataset['categories']:
        class_count[cat['id']] = [0, 0 , 0, 0]  # all,ignore, areafilter, groundtruth


    for ann in dataset['annotations']:
        cat_id  = ann['category_id']
        class_count[cat_id][0]+=1
        if ann['ignore']:
            class_count[cat_id][1] += 1
        elif int(ann['area']) < area_filter:
            class_count[cat_id][2] += 1
        else:
            class_count[cat_id][3] += 1

    return class_count

def selected_class_from_coco_file(annotation_file, class_dict):
    """
    select annotation info of new category from coco file
    :param annotation_file:  origin coco file
    :param class_dict: selected class dict -> dict
    :return:
        dataset: annotation info
    """
    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    annotation = []
    images = []
    img_id_map = {}
    id_c = 0
    image_id = 0
    class_before = convert_catgories2list(dataset['categories'])

    for ann in dataset['annotations']:
        before_name = class_before[ann['category_id']-1]
        if before_name in class_dict.keys():
            id_c += 1
            # ann['category_id'] = class_list.index(before_name) + 1
            ann['category_id'] = class_dict[before_name]
            ann['id'] = id_c
            if ann['image_id'] not in img_id_map.keys():
                image_id += 1
                img_id_map[ann['image_id']] = image_id
            ann['image_id'] = img_id_map[ann['image_id']]
            annotation.append(ann)
    for image in dataset['images']:
        if image['id'] in img_id_map.keys():
            image['id'] = img_id_map[image['id']]
            images.append(image)

    dataset['annotations'] = annotation
    dataset['images'] = images
    dataset['categories'] = convert_dict2catgories(class_dict)
    return dataset

def selected_data_from_old(origin_annotation, origin_pic, save_dir, mode,class_dict):
        annotation_file = os.path.join(origin_annotation,  f'COCO/annotation/{mode}.json')
        json_save_path = os.path.join(save_dir, 'COCO', 'annotation')
        pic_save_path = os.path.join(save_dir, mode)
        mkdir(json_save_path)
        mkdir(pic_save_path)
        # dota2
        # 原始数据的类别
        # class_before = ('small-vehicle', 'tennis-court', 'large-vehicle', 'baseball-diamond', 'ship',
        #    'ground-track-field', 'basketball-court', 'harbor', 'bridge', 'roundabout', 'soccer-ball-field',
        #    'swimming-pool', 'airport', 'storage-tank', 'plane', 'helipad', 'helicopter', 'container-crane')
        # class_list = ['ship', 'plane', 'helicopter']
        # 要筛选的数据类别，及想要赋予的id
        # class_dict = {'small-vehicle':1 , 'large-vehicle':2 , 'ship':3 ,
        #               'harbor':4 , 'bridge':5 ,
        #               'airport':6 , 'plane':7 , 'helipad':8 , 'helicopter':9}

        data_dict = selected_class_from_coco_file(annotation_file, class_dict)

        data_dict['info'] = {'contributor': 'XeatherH',
                           'data_created': datetime.now().strftime("%Y.%m.%d"),
                           'description': f'This is  a selected dataset from DOTA2.0, select category of:{class_dict.keys()}',
                           'url': None,
                           'version': '1.0'}

        with open(json_save_path + f'/{mode}.json', 'w') as f:
            json.dump(data_dict, f, indent=4)
        with open(json_save_path + f'/{mode}.txt', 'w') as t:
            for img in tqdm(data_dict['images']):
                t.write(img['file_name']+'\n')
                # 将筛选后的图片复制放入
                origin_pic_path = origin_pic+f'/{mode}/images/'+img['file_name']
                new_pic_path = pic_save_path+'/'+img['file_name']
                shutil.copy2(origin_pic_path, new_pic_path)

        log.info(f'{mode}.json and txt is saved in {json_save_path}')

def picfile_txt(save_path, origin_pic_path, mode_l=('train', 'val'), data_split=(0.8, 0.2)):
    """
    we random split the dataset to mode list, and write the file_name to mode list
    :param save_path:
    :param origin_pic_path: could be list or tuple, means we select pic from more than one dataset
    :param model:
    :param data_split:
    :return:
    """
    assert len(mode_l)==len(data_split), 'data split length should be same with data mode length'
    data_split = np.array(data_split)*100//data_split.sum().tolist()
    data_split.insert(0, 0)
    rand_num_list = [data_split[:i+1].sum() for i in range(len(data_split))]
    rand_num_list[-1] = 100
    pic_suffix = ['png','jpg','jpeg','tiff']
    file_in=[]
    for mode in mode_l:
        file_in.append(open(f'{save_path}/{mode}.txt', 'w'))
    if not isinstance(origin_pic_path,(list,tuple)):
        origin_pic_path = (origin_pic_path)
    for pic_path in origin_pic_path:
        for img in tqdm(glob(pic_path+'/*.*')):
            if img.split('.')[-1] in pic_suffix:
                num = random.randint(1, 100)
                for i, file in enumerate(file_in):
                    if num >= rand_num_list[i] and num < rand_num_list[i+1]:
                        file.write(img+'\n')
    for f in file_in:
        f.close()

def calcu_yolo_txt(label_path, cate):
    classes = []
    # cate = {'vehicle': 1, 'bridge': 2, 'ship': 3,
    #         'airport': 4, 'harbor': 5, 'airplane': 6, 'helipad': 7, 'helicopter': 8}
    txt_files = os.listdir(label_path)
    for file in txt_files:

        path1 = os.path.join(label_path, file)
        data = np.loadtxt(path1)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]

        cls = data[:, 0: 1].flatten().tolist()
        classes.extend(cls)

    classes = np.array(classes)
    cat_yolo_dict = {}
    if isinstance(cate, list):
        for per_cat in cate:
            cat_yolo_dict[per_cat['id']-1] = per_cat['name']
    else:
        cat_yolo_dict = cate
    for cat_id, name in cat_yolo_dict.items():
        print(f'{name} count is {sum(classes == cat_id)}')

def convert_coco_json(root_dir='.',coco_path='COCO/annotation', use_segments=False, mode='test',area_filter=None, str_filter=''):
    """
    convert json 2 yolo txt labels and write the image_file_path to mode.txt in label path.
    :param root_dir:
    :param coco_path:
    :param use_segments:
    :param mode:
    :param area_filter: filter the box area
    :param str_filter: filter the image file name
    :return:
    """
    json_file = os.path.join(root_dir, coco_path, f'{mode}.json')
    if not os.path.exists(json_file):
        return 0

    # 防止已转换完成后，修改信息后再次转换的重复叠加(open txt 为 a 模式)
    if os.path.exists(os.path.join(root_dir, 'labels', mode)):
        shutil.rmtree(os.path.join(root_dir, 'labels', mode))
    object_path = os.path.join(root_dir, 'images', mode, 'object')
    if os.path.exists(object_path):
        img_path = object_path
        save_dir = os.path.join(root_dir, 'labels', mode, 'object')
    else:
        img_path = os.path.join(root_dir, 'images', mode)
        save_dir = os.path.join(root_dir, 'labels', mode)
    file_names = []
    mkdir(save_dir)
    # Import json

    fn = Path(save_dir)  # folder name
    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
        if x['ignore']:
            continue
        if area_filter is not None:
            if int(x['area'])<area_filter:
                continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']
        if check_platform()=='Windows':
            f = f.replace(':', '_')
        if contain_str(f, str_filter):
            continue
        file_names.append(f)

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Segments
        if use_segments:
            segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

        # Write
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = int(x['category_id']) - 1  # class
            line = cls, *(s if use_segments else box)  # cls, box or segments
            # 设置为a,是添加写入，x是依照id,读取，不同x imageid可能为同一个，故为同一张图片，故需要多次打开存入
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

    if os.path.exists(img_path):
        with open(os.path.join(root_dir, 'labels', f'{mode}.txt'), 'w') as f:
            file_names =  set(file_names)
            for file in file_names:
                f.write(os.path.join(img_path, file) + '\n')

    calcu_yolo_txt(save_dir,data['categories'])

    return 1

def get_box_from_xml(xmlfile, classes=None):
    object_axis = []
    xml_class = []
    in_file = open(xmlfile)
    tree=ET.parse(in_file)
    root = tree.getroot()
    # for child1 in root:
    #     for child2 in child1:
    #         if child2.tag == 'width':
    #             width = child2.text
    for size in root.iter('size'):
        width = float(size.find('width').text)
        height = float(size.find('height').text)
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = obj.find('Difficult').text
        cls = obj.find('name').text.lower()
        if classes is not None:
            if cls not in classes:
                continue
        else:
            xml_class.append(cls)
        # if int(difficult) == 1:
        #     continue
        xmlbox = obj.find('bndbox')
        box_trans = True
        box_unspec = False
        pose = obj.find('pose')
        if pose is not None:
            pose = pose.text
            if pose=='FourSpot':
                box_trans = False
            elif pose=='Unspecified':
                box_unspec = True

        if box_unspec:
            #assert xmlbox.find('x0') is not None, f'in {xmlfile} box is not identified'
            if xmlbox.find('x0') is not None:
                b = [float(xmlbox.find('x0').text), float(xmlbox.find('y0').text),
                     float(xmlbox.find('x1').text), float(xmlbox.find('y2').text),
                     cls, int(difficult)]
            elif xmlbox.find('xmin') is not None:
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('xmax').text),float(xmlbox.find('ymax').text),
                     cls, int(difficult)]
            else:
                assert xmlbox.find('x0') is not None, f'in {xmlfile} box is not identified'

        elif box_trans:
            b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymax').text), cls, int(difficult)]
        else:
            b = [float(xmlbox.find('xmin').text.split('-')[0]), float(xmlbox.find('ymin').text.split('-')[0]), float(xmlbox.find('xmax').text.split('-')[0]),
                 float(xmlbox.find('ymax').text.split('-')[0]), cls, int(difficult)]
        object_axis.append(b)
    return (object_axis,xml_class) if classes is None else object_axis, height, width

def convert_dict2catgories(dict):
    cats = []
    for name,id in dict.items():
        cat = { "id": id,
            "name": name,
            "supercategory": name}
        cats.append(cat)
    return cats
def convert_catgories2dict(cats):
    dict = {}
    for cat in cats:
        name = cat["name"]
        id = cat["id"]
        dict[name] = id
    return dict

def convert_catgories2list(cats):
    list_c = [0]*80
    for cat in cats:
        name = cat["name"]
        id = cat["id"]
        list_c[id-1]=name
    return list_c

def labels_to_class_weights(labels, nc=80, empty_bins=1e16):

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = empty_bins  # replace empty bins with 10**16
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights

def small_sample_from_coco(origin_dir,mod,percent=0.1,newanno_prefix='select_',save_pic=False, cat_select=False, filter_str=''):

    # parent_path = os.path.dirname(os.path.dirname(os.path.abspath(annotation_file)))
    annotation_dir = origin_dir + '/COCO/annotation/'
    annotation_file = annotation_dir+f'{mod}.json'

    if save_pic:
        pic_ori_dir = os.path.dirname(origin_dir)+f"/images/{mod}/"
        pic_save_dir = os.path.dirname(origin_dir) + f"/images/{newanno_prefix}{mod}/"
        mkdir(pic_save_dir, rm=True)


    with open(annotation_file, 'r') as f:
        dataset = json.load(f)
    small_anno = annotation_dir + f'{newanno_prefix}{mod}.json'

    info = {'contributor': 'XeatherH',
            'data_created': datetime.now().strftime("%Y.%m.%d"),
            'description': f'random select to {percent} pictures from dataset of {origin_dir} and filter string is {filter_str}',
            'url': None,
            'version': '1.0'}
    images = []
    annotations = []
    new_id = 1
    obj_id = 1
    if isinstance(percent, int):
        max_pic_num = percent
        percent = 2 * percent/len(dataset['images'])
    else:
        max_pic_num = len(dataset['images'])

    # if cat_select:
    #     cw = labels_to_class_weights(labels, nc)
    #     iw = labels_to_image_weights(labels, nc=nc, class_weights=cw)  # image weights
    #     add_indices = random.choices(range(img_num), weights=iw, k=img_num)  # rand weighted idx
    #

    for pic_info in tqdm(dataset['images']):
        rand = random.randint(0, 10000)
        if contain_str(pic_info['file_name'], filter_str=filter_str):
            continue
        if rand < 10000 * percent:
            image_id = pic_info['id']
            pic_info['id'] = new_id
            images.append(pic_info)
            if save_pic:
                pic_file = pic_info['file_name']
                shutil.copy(pic_ori_dir+pic_file, pic_save_dir+pic_file)
            for anno_info in dataset['annotations']:
                if anno_info['image_id'] == image_id:
                    anno_info['image_id'] = new_id
                    anno_info['id'] = obj_id
                    obj_id += 1
                    annotations.append(anno_info)
                if np.array(anno_info['segmentation']).ndim < 2:
                    anno_info['segmentation'] = [anno_info['segmentation']]
            new_id += 1
        if new_id > max_pic_num:
            # print(images)
            break


    data_dict = {'info': info, "images": images, "type": "instances", "annotations": annotations,
                      "categories": dataset['categories']}
    with open(small_anno, 'w') as t:
        json.dump(data_dict, t, indent=4)

def contain_str(orig_str, filter_str):
    if filter_str is not None and filter_str != '':
        if isinstance(filter_str, str):
            filter_str = [filter_str]
        for f_str in filter_str:
            if f_str in orig_str:
                return True
    return False

def select_pic(path, save_dir, percent=0.2, filter_str=None):
    mkdir(save_dir, rm=True)

    if percent < 1:
        percent *= 100
    for p in tqdm(glob(path + '/*.*')):
        file_name = os.path.split(p)[-1]
        if contain_str(file_name, filter_str):
            continue
        a = random.randint(1,100)
        if a<=percent:
            shutil.copy(p, os.path.join(save_dir, file_name))

def select_label(pic_path, label_path, save_dir):
    mkdir(save_dir, rm=True)
    for p in tqdm(glob(pic_path + '/*.*')):
        file_name = os.path.split(p)[-1].replace('.png', '.txt').replace('.jpg', '.txt')
        try:
            shutil.copy(os.path.join(label_path, file_name),os.path.join(save_dir,file_name))
        except:
            with open(os.path.join(save_dir, file_name), 'w') as f:
                f.write('')

def summary_label2txt(label_path, suffixs=('.png', '.jpg')):
    f = open(label_path+'.txt', 'w+')
    for p in tqdm(glob(label_path + '/*.*')):
        dir, file_name = os.path.split(p)
        for suffix in suffixs:
            pic_name = file_name.replace('.txt', suffix)

            pic_path = dir.replace('labels', 'images') + '/' + pic_name
            if os.path.exists(pic_path):
                f.write(pic_path+'\n')
    f.close()

def select_pic_from_cat(path, mode, save_mode, cate, percent=0.2, cate_select=None, suffixes=['.jpg', '.png']):
    if percent < 1:
        percent *= 100
    pic_path = os.path.join(path, 'images', mode)
    save_pic = os.path.join(path, 'images', save_mode)
    label_path = os.path.join(path, 'labels', mode)
    save_label = os.path.join(path, 'labels', save_mode)
    mkdir(save_pic, rm=True)
    mkdir(save_label, rm=True)

    for p in tqdm(glob(label_path + '/*.*')):
        label_file_name = os.path.split(p)[-1]
        pic_file_name_ = label_file_name.replace('.txt', '')
        data = np.loadtxt(p)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        cls = data[:, 0: 1].flatten().tolist()
        if cate_select is not None:
            for cat in cate_select:
                if cat in cls:
                    rand_num = random.randint(1,100)
                    if rand_num <= percent:
                        shutil.copy(p, os.path.join(save_label, label_file_name))
                        for suffix in suffixes:
                            pic_file_name = pic_file_name_ + suffix
                            pic_file = os.path.join(pic_path, pic_file_name)
                            if os.path.exists(pic_file):
                                shutil.copy(pic_file, os.path.join(save_pic, pic_file_name))
                                break
                    break
    calcu_yolo_txt(save_label, cate)


def pic_wnum(path, save_path,txt_color=(0, 0, 0), mode='', label_path=None, suffixs='txt', i=0):
    if label_path is None:
        label_path = os.path.join(path, 'labels', mode)
    mkdir(save_path)
    mkdir(os.path.join(save_path, 'images', mode))
    mkdir(os.path.join(save_path, 'labels', mode))
    if isinstance(suffixs, str):
        suffixs = [suffixs]
    for p in tqdm(glob(os.path.join(path, 'images', mode) + '/*.*')):
        img = cv2.imread(p)
        cv2.putText(img, str(i+1), (10, 30), 0, 1, txt_color,
                    thickness=2, lineType=cv2.LINE_AA)
        file_name = os.path.split(p)[-1]
        cv2.imwrite(os.path.join(save_path, 'images', mode, str(i+1).zfill(5)+'.jpg'), img)
        for suffix in suffixs:
            org_name = file_name.replace('.png', suffix).replace('.jpg', suffix)
            shutil.copy(os.path.join(label_path, org_name),
                        os.path.join(save_path, 'labels', mode, str(i + 1).zfill(5) + suffix))
            if suffix in ['.xml', '.XML']:
                xml_f = os.path.join(save_path, 'labels', mode, str(i + 1).zfill(5) + suffix)

                tree = ET.parse(xml_f)
                root = tree.getroot()
                fname = root.find('filename')
                fname.text = str(i+1).zfill(5)+'.jpg'
                tree.write(xml_f)
        i+=1

def bgr2gray(path,save_dir):
    mkdir(save_dir)
    for p in tqdm(glob(path + '/*.*')):
        img = cv2.imread(p, 2)
        img_dir, file_name = os.path.split(p)
        cv2.imwrite(os.path.join(save_dir,file_name), img)


def convert_annotation(xmlfile, list_file, classes):
    object_axis, height, width = get_box_from_xml(xmlfile, classes)

    for obj in object_axis:
        x1, y1, x2, y2, cls = obj
        cls_id = classes.index(cls)
        c =( (x2+x1)/2/width, (y2+y1)/2/height, (x2-x1)/width, (y2-y1)/height )
        list_file.write(str(cls_id)+' '+" ".join([str(a) for a in c])+'\n')

# 将类别名字和id建立索引
def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes


# 将标签信息写入xml
def save_anno_to_xml(filename, size, objs, save_path):
    # from xml import objectify
    def objectify():
        pass
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder("DATA"),
        E.filename(filename),
        E.source(
            E.database("The VOC Database"),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size['width']),
            E.height(size['height']),
            E.depth(size['depth'])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose("Unspecified"),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[1]),
                E.ymin(obj[2]),
                E.xmax(obj[3]),
                E.ymax(obj[4])
            )
        )
        anno_tree.append(anno_tree2)
    anno_path = os.path.join(save_path, filename[:-3] + "xml")
    ET(anno_tree).write(anno_path, pretty_print=True)


# 利用cocoAPI从json中加载信息
def coco_2xml(anno_file, xml_save_path):
    if os.path.exists(xml_save_path):
        shutil.rmtree(xml_save_path)
    os.makedirs(xml_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()
    for imgId in tqdm(imgIds):
        size = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        width = img['width']
        height = img['height']
        size['width'] = width
        size['height'] = height
        size['depth'] = 1
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            # bbox:[x,y,w,h]
            bbox = list(map(int, ann['bbox']))
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            obj = [object_name, xmin, ymin, xmax, ymax]
            objs.append(obj)
        save_anno_to_xml(filename, size, objs, xml_save_path)
def format_label(txt_list,class_dict=None):
    format_data = []
    for i in txt_list:
        i = i.strip()
        i_list = i.split(' ')
        if len(i_list) < 5:
            continue
        class_name = i_list[0]
        if class_dict is not None:
            class_name = class_dict[int(class_name)]

        difficult = '0'

        if len(i_list) == 6:
            difficult = i_list[-1]

        box_id = list(map(float, i_list[1:5]))
        box_id.append(class_name)
        box_id.append(difficult)
        format_data.append(box_id)
    return format_data
def yolo_to_voc(raw_images_dir, save_path, class_dict=None):
    images = [i for i in os.listdir(raw_images_dir) if 'png' or 'jpg' in i]
    raw_label_dir = raw_images_dir.replace('images', 'labels')
    # labels = [i for i in os.listdir(self.raw_label_dir) if 'txt' in i]
    # print('find image', len(images))
    # print('find label', len(labels))
    for idx, img in enumerate(tqdm(images)):
        # print(idx, 'read image', img)
        label_path = os.path.join(raw_label_dir, img.replace('png', 'txt').replace('jpg', 'txt'))
        if not os.path.exists(label_path):
            continue

        txt_data = open(label_path, 'r').readlines()
        box = format_label(txt_data, class_dict)

        pic_name = img
        pic_path = os.path.join(raw_images_dir, img)
        pic_data = cv2.imread(pic_path)
        height, width = pic_data.shape[:2]
        for b in box:
            w = b[2] * width
            h = b[3] * height
            cx = b[0] * width
            cy = b[1] * height
            b[0] = int(cx - w/2)
            b[1] = int(cy - h/2)
            b[2] = int(cx + w/2)
            b[3] = int(cy + h/2)

        assert pic_data is not None, f'{pic_name} is nonetype ,{pic_data}'
        mkdir(save_path)
        xml = os.path.join(save_path, pic_name.replace('png', 'xml'))
        size = {}
        size['width'] = width
        size['height'] = height
        size['depth'] = 1
        save_to_xml(pic_name, xml, size, box)

# ## yolotxt2coco
# def save_to_json(self, image_id, pic_name, save_path, im_height, im_width, objects_axis):
#     if image_id == 1:
#         class_dict = {}
#         self.class_map_list = self.class_map_return(save_path)
#         info = {'contributor': 'captain group',
#                 'data_created': '2021',
#                 'description': 'This is 2.0 version of DOTA dataset.',
#                 'url': None,
#                 'version': '2.0',
#                 'year': 2021}
#
#         self.data_dict = {'info': info, "images": [], "type": "instances", "annotations": [],
#                      "categories": []}
#
#     image = {
#         'file_name': pic_name,
#         'height': im_height,
#         'width': im_width,
#         'id': image_id
#     }
#     self.data_dict['images'].append(image)
#     for box in objects_axis:
#         obj_box = box[:8]
#         poly = [(obj_box[2*i], obj_box[2*i+1]) for i in range(4)]
#         gtpoly = shgeo.Polygon(poly)
#         area = gtpoly.area
#         xmin, ymin, xmax, ymax = min(obj_box[0::2]), min(obj_box[1::2]), \
#                                  max(obj_box[0::2]), max(obj_box[1::2])
#         o_width = xmax - xmin
#         o_height = ymax - ymin
#         self.bnd_id += 1
#         category_id = self.class_map_list.index(box[8]) + 1
#         ann = {'area': area, 'iscrowd': 0, 'image_id': image_id, 'bbox': [xmin, ymin, o_width, o_height],
#                'category_id': category_id, 'id': self.bnd_id, 'ignore': 0,
#                'segmentation': [obj_box]}
#         self.data_dict['annotations'].append(ann)

def calculate_box(length, distance, f, dpi, fov):
    """
    calculate the width in the picture
    :param length: length of real object
    :param distance: the distance to object
    :param f: focus distance of camera lens
    :param dpi: camera pixel
    :param fov: field of view
    :return:
        obj_width: length of  object in the picture
    """
    # fov = 2 * atan( W / (2*f) ), W means ccd靶面规格尺寸
    W = 2*f*math.tan(fov/2)
    print('target width is ', W)
    # f/d = (W/ dpi * obj_width) / length

    obj_width = length * dpi /(2*distance*math.tan(fov/2))
    print("object width in the picture is ", obj_width)

    return obj_width


if __name__ == '__main__':
    length = 3*1e3  # mm
    distance = 10*1e6 # mm
    fs = [22, 720, 1050]

    print("start calcu width of object")
    fovs = [24.6, 0.76, 0.52]
    dpi = 640

    for f, fov in zip(fs, fovs):
        # fov angle to rad
        fov = fov / 180 * math.pi
        calculate_box(length, distance, f, dpi, fov)

    print("start calcu height of object")
    fovs = [19.8, 0.61, 0.41]
    dpi = 512
    for f, fov in zip(fs, fovs):
        # fov angle to rad
        fov = fov / 180 * math.pi
        calculate_box(length, distance, f, dpi, fov)
    # length = 10*1e3  # mm
    # distance = 50*1e6 # mm
    # fs = [13, 255, 550]
    # fovs = [28, 1.5, 0.9]
    # dpi =1920
    # for f, fov in zip(fs, fovs):
    #     fov = fov / 180 * math.pi
    #     calculate_box(length, distance, f, dpi, fov)

