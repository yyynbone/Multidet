# from utils import plot_results
# if __name__ == '__main__':
#     csv_file = '../results/train/merge_scls/yolo_classify/exp/result.csv'
#     plot_results(csv_file)

# from loss import CrossEntropyLoss, binary_cross_entropy
# import torch
# if __name__ == '__main__':
#     loss_f = CrossEntropyLoss(use_sigmoid=True)
#     pred  = torch.tensor([[0.9],[0.1],[0.1], [0.9],[0.1],[0.1]])
#     label = torch.tensor([[1.], [1.], [0], [0.], [0.], [0]])
#     loss = loss_f.forward(pred, label)
#     print(pred.sigmoid())
#     print(loss)
#     loss = loss*torch.tensor([[10.], [10.], [1], [1.], [1.], [1]])
#     print(loss)
#     loss =loss.sum()/24
#     print(loss)
#
#     # pred  = torch.tensor([[0.9],[0.5],[0.5]])
#     # label = torch.tensor([[1.], [1.], [0]])
#     # loss = loss_f.forward(pred, label)
#
#     loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([10.], device=pred.device))
#     print((loss))
#     loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([1000.], device=pred.device))
#     print((loss))
#     print("########################")
#     pred  = torch.tensor([[0.9],[0.1],[0.5], [0.9],[0.1],[0.1]])
#     label = torch.tensor([[1.], [1.], [1.], [0.], [0.], [0]])
#     loss = loss_f.forward(pred, label)
#     print(pred.sigmoid())
#     print(loss)
#
#     loss = loss*torch.tensor([[10.], [10.], [10.], [1.], [1.], [1]])
#     loss =loss.sum()/33
#     print(loss)
#     loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([10.], device=pred.device))
#     print((loss))
#     loss = binary_cross_entropy(pred, label, pos_weight=torch.tensor([1000.], device=pred.device))
#     print((loss))


