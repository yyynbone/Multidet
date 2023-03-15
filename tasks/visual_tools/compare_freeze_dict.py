
from utils import show_model_param, show_state_dict
import torch
import cv2
if __name__ == '__main__':
    a = '../../results/train/merge_scls/yolo_classify/exp4/weights/best_precision.pt'
    b = '../../results/train/merge_scls/yolo_classify/exp12/weights/best.pt'
    model_a = torch.load(a, map_location='cuda:0')
    model_b = torch.load(b, map_location='cuda:0')
    # show_state_dict(model_a['model'].state_dict(), file='a_dict.json', dict_idx=list(range(10)))  # contains bn.running_mean bn.running_var which cant freeze
    # show_state_dict(model_b['model'].state_dict(), file='b_dict.json', dict_idx=list(range(10)))
    # show_model_param(model_a['model'],file='a_param.json', layer_idx=10)
    # show_model_param(model_b['model'],file='b_param.json', layer_idx=10)
    pic_file = '../../images/bus.jpg'
    img = cv2.imread(pic_file)
    input =  torch.tensor(img[..., 0][None,None,...]/255., device='cuda:0').half()
    m_a = model_a['model'].model[0]
    m_b = model_b['model'].model[0]
    # # print(m_a)
    # a_o = m_a(input)
    # # print(m_b)
    # b_o = m_b(input)
    # # print(torch.subtract(m_a(input), m_b(input))) # 输出完全不一样，说明runing_mean_bn 是参与计算的
    # print('before eval compare', torch.abs((a_o - b_o)).sum())  # inf
    ######## 比较model.eval() 后###########
    m_a.eval()
    a_o_e = m_a(input)
    m_b.eval()
    b_o_e = m_b(input)
    print(a_o_e, b_o_e)
    print('after eval compare', torch.abs((a_o_e - b_o_e)).sum())  # inf
    # print('not eval and eval compare of a', torch.abs((a_o - a_o_e)).sum()) # 0


