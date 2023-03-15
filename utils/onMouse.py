# encoding:utf8
import cv2
import numpy as np

new_box = []
def onMouse(event, x, y, flags, userdata):
    # 右键单击获得修正框  new_box = [xmin,ymin,xmax,ymax]
    if event == cv2.EVENT_RBUTTONDOWN :
        new_box.extend((x,y))
        print(new_box)
    # 左键单击
    if event == cv2.EVENT_LBUTTONDOWN :
        print(x,y)

def check_box(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('raw')                                 
    # 鼠标事件
    cv2.setMouseCallback('raw', onMouse)
    cv2.imshow('raw', img)
    cv2.waitKey(0)
    cv2.destroyWindow('raw')
    if len(new_box) >= 4:
        boxs = np.array(new_box).reshape(-1,4)
        for box in boxs:
            cv2.rectangle(img, (box[0], box[1]),(box[2], box[3]), (0, 255, 0), 3)
        cv2.imshow('new',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    img_path = r'E:\workspace\data_all\22-FAIR1M\images\1.tif'
    check_box(img_path)