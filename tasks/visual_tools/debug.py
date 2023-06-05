# from utils import plot_results
# if __name__ == '__main__':
#     csv_file = '../results/train/merge_scls/yolo_classify/exp/result.csv'
#     plot_results(csv_file)

from loss import CrossEntropyLoss, binary_cross_entropy
import torch
from torch import nn
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import contextmanager

def dist_test(local_rank, node=0, local_size=8, addr='localhost', port=55555, world_size=8):
    i = 0
    torch.cuda.set_device(local_rank)
    # device = torch.device('cuda', local_rank)
    Rank = local_rank + node * local_size
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                            init_method="tcp://{}:{}".format(addr, port),
                            rank=Rank,
                            world_size=world_size)
    i = dist_barrier(local_rank, i)
    print("####################################")
    if local_rank in [-1, 0]:
        dist_barrier(-1, 5)

def torch_distributed_zero_first(local_rank):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        print('torch dist', local_rank)
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        print('torch dist', local_rank)
        dist.barrier(device_ids=[0])

# @contextmanager
# def torch_distributed_zero_first(local_rank: int):
#     """
#     Decorator to make all processes in distributed training wait for each local_master to do something.
#     """
#     if local_rank not in [-1, 0]:
#         print('torch dist', local_rank)
#         dist.barrier(device_ids=[local_rank])
#     yield #
#     if local_rank == 0:
#         print('torch dist', local_rank)
#         dist.barrier(device_ids=[0])



def yeildfunc():
    """
    function use yield， must use next() or send() to call the func
        use：
            a = next(yeildfunc())
            >: now 1
            yeildfunc()  # function not called
            >:
    :return:
    """
    print('now 1')
    yield
    print('now 2')

def dist_barrier(rank, i):
    # with torch_distributed_zero_first(rank):
    #     i += 1
    #     time.sleep(2)
    #     print(f'now we are in  {rank},  i is {i}')
    #     return i
    # next(torch_distributed_zero_first(rank))
    # i += 1
    # time.sleep(2)
    # print(f'now we are in  {rank},  i is {i}')
    # return i

    # if rank not in [-1, 0]:
    #     print('torch dist', rank)
    #     dist.barrier(device_ids=[rank])

    try:
        i += 1
        # time.sleep(2*(rank+2))
        print(f'now we are in  {rank},  i is {i}')
        return i

    finally:
        print(f"now in  {rank}")
        # if rank == 0:
        #     print('torch dist', rank)
        #     dist.barrier(device_ids=[0])


if __name__ == '__main__':
    # loss_f = CrossEntropyLoss(use_sigmoid=True)
    # pred  = torch.tensor([[0.9],[0.1],[0.1], [0.9],[0.1],[0.1]])
    # label = torch.tensor([[1.], [1.], [0], [0.], [0.], [0]])
    # loss = loss_f.forward(pred, label)
    # print(pred.sigmoid())
    # print(loss)
    # loss = loss*torch.tensor([[10.], [10.], [1], [1.], [1.], [1]])
    # print(loss)
    # loss =loss.sum()/24
    # print(loss)
    #
    # # pred  = torch.tensor([[0.9],[0.5],[0.5]])
    # # label = torch.tensor([[1.], [1.], [0]])
    # # loss = loss_f.forward(pred, label)
    #
    # loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([10.], device=pred.device))
    # print((loss))
    # loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([1000.], device=pred.device))
    # print((loss))
    # print("########################")
    # pred  = torch.tensor([[0.9],[0.1],[0.5], [0.9],[0.1],[0.1]])
    # label = torch.tensor([[1.], [1.], [1.], [0.], [0.], [0]])
    # loss = loss_f.forward(pred, label)
    # print(pred.sigmoid())
    # print(loss)
    #
    # loss = loss*torch.tensor([[10.], [10.], [10.], [1.], [1.], [1]])
    # loss =loss.sum()/33
    # print(loss)
    # loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([10.], device=pred.device))
    # print((loss))
    # loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([1000.], device=pred.device))
    # print((loss))

    # pred  = torch.tensor([[0.5],[0.1],[0.3], [0.7],[0.1],[0.7]])
    # label = torch.tensor([[1.], [1.], [0], [0.], [0.], [0]])
    # loss =  torch.nn.functional.binary_cross_entropy(pred, label,reduction='none')
    # pt = (1 - pred) * label + pred * (1 - label)
    # pt_pw = pt.pow(2)
    # pos_weight = torch.tensor([1.])
    # alpha = pos_weight/ (1+pos_weight)
    # bias = (alpha * label + (1 - alpha) *
    #                 (1 - label))
    # focal_weight = bias * pt_pw
    # print(pt, pt_pw, bias)
    # print(loss, focal_weight)
    # print(loss*focal_weight)



    # import numpy as np
    # from dataloader import mask_label
    # import cv2
    #
    # result = {
    #     'filename': 'D:\\gitlab\\trainingsys\\data\\essential\\merged_of_auto_resized_iou48_800_800dota_and_dior\\images\\val\\dior_00872-0_800_0_800.jpg',
    #     'ori_shape': (800, 800),
    #     'img_size': [256, 256],
    #     'labels': np.array([[0., 342., 483., 429., 547.],
    #                         [0., 395., 246., 464., 319.],
    #                         [0., 437.00003, 472.00003, 523., 534.],
    #                         [0., 463., 159.99998, 550., 227.],
    #                         [0., 534., 458.99997, 618., 519.],
    #                         [0., 674., 122., 739., 202.],
    #                         [0., 679., 45.999996, 741., 108.99999]])}
    # result['img'] = cv2.imread(result['filename'])
    # mask_label(result)


    mp.spawn(dist_test,
             args=(),
             nprocs=8,
             join=True)



