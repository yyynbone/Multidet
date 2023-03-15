# encoding:utf8
import copy
import json
import cv2
import numpy as np
import os

from tqdm import tqdm
from crop_coco_tools.utils import get_pic_id_map
from datetime import datetime

def onMouse(event, x, y, flags, param):
    # 右键单击获得修正框  new_box = [xmin,ymin,xmax,ymax]
    if event == cv2.EVENT_RBUTTONDOWN and len((param[0])) % 6 !=0:
        param[0].extend((x,y))
        print(f'修正坐标:{x,y}')
        if len(param[0]) % 6 == 0:
            cv2.rectangle(param[1], (param[0][-4], param[0][-3]),(param[0][-2], param[0][-1]), (0, 255, 0), 2)
            print(f'修改成功，修正框信息:{param[0][-4], param[0][-3],param[0][-2]-param[0][-4], param[0][-1]-param[0][-3]}')
            print('输入回车继续添加修正框信息!(左上为目标序号,左下为目标类别)')
            print('输入空格查看下一张图片!')
            print('输入ESC终止操作!')
    # 左键单击显示坐标
    if event == cv2.EVENT_LBUTTONDOWN :
        print(f'当前坐标:{x,y}')
def re_label(ann,box0):
    if box0:
        box0 = np.array(box0).reshape(-1,6)
        boxw = box0[:,4:5] - box0[:,2:3]
        boxh = box0[:,5:6] - box0[:,3:4]
        area = boxw * boxh
        box = np.concatenate((box0[:,0:1],box0[:,2:4],boxw,boxh,area,box0[:,1:2]),1)
        for b in box:
            if b[0] < len(ann):
                ann[b[0]] = b[1:].tolist()
            else:
                ann.append(b[1:].tolist())
    return ann 
def check_box(anno_data,img_path,new_json_path,save_json = False):
    """ 
    save_json 为 True 则保存新的 json 文件
    """
    print('输入回车添加修正框信息!(左上为目标序号,左下为目标类别)')
    print('输入空格查看下一张图片!')
    print('输入ESC终止操作!')
    for ann in anno_data['anno_info']:        
        file_name = os.path.join(img_path,ann['file_name'])
        img0 = cv2.imread(file_name)
        img1 = copy.deepcopy(img0)
        new_box = []
        cv2.namedWindow('raw')
        cv2.setMouseCallback('raw', onMouse,(new_box,img1))
        for i,box in enumerate(ann['anno_info']):
            cv2.rectangle(img1,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),2)
            cv2.putText(img1, str(i), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 左上角 idx        
            cv2.putText(img1, str(box[-1]), (box[0], box[1] + box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 右下角 class        
        while True:
            cv2.imshow('raw', img1)
            key = cv2.waitKey(30)
            # 回车输入需要修正目标信息
            if key == 13:               
                idx = int(input('请输入预修正目标的序号: '))
                cls = int(input('请输入预修正目标的类别: '))
                new_box.extend((idx,cls))
                print(f'目标序号和类别:{new_box[-2],new_box[-1]}')
                print('右键点击添加修正框左上坐标及右下坐标!')              
            # 空格检查下一张图片
            if key == 32:
                ann['anno_info'] = re_label(ann['anno_info'],new_box)
                break
            # ESC退出程序
            if key == 27:
                ann['anno_info'] = re_label(ann['anno_info'],new_box)
                print('图片审核已终止!')
                return
        cv2.destroyAllWindows
    if save_json:
        data2json(anno_data,new_json_path)
def data2json(anno_data,json_path):
    json_data = {}    
    json_data['images'] = []
    json_data['annotations'] = []    
    img_id = 0    
    ann_id = 0    
    json_data['info'] = anno_data['info']
    json_data['categories'] = anno_data['categories']   
    for img_ann in tqdm(anno_data['anno_info']):        
        img_id += 1
        img_info = {
            'file_name':img_ann['file_name'],
            'height':img_ann['height'],
            'width': img_ann['width'],
            'id':img_id
        }
        json_data['images'].append(img_info)
        for ann in img_ann['anno_info']:
            ann_id += 1
            ann_info = {
                "area": ann[4],
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [
                    ann[0],
                    ann[1],
                    ann[2],
                    ann[3],                        
                ],
                "category_id": ann[5],
                "id": ann_id,
                "ignore": 0,
                "segmentation": [
                    []
                ]
            }
            json_data['annotations'].append(ann_info)
    # if not os.path.exists(json_path):
    #     os.makedirs(json_path)
    with open((json_path),'w') as f:
        json.dump(json_data,f,indent=4)    
if __name__ == '__main__':
    json_path = r'E:\workspace\test_data\DOTA100\x_all.json'                # json 原文件
    new_json_path = r'E:\workspace\test_data\DOTA100\new_all.json'          # json 修改后文件保存路径
    img_path = r'E:\workspace\test_data\DOTA100\imgtest'                    # 图片路径
    anno_data, _ = get_pic_id_map(json_path)                                # 获得 anno_data
    check_box(anno_data,img_path,new_json_path,save_json=False)             # 半自动监督生成 new_json_data,save_json是否保存为新的 json 文件
                        
    