# edit by xeatherH ,2022 / 7 /22
# if grad in x epoch with x epoch loss, it is  easy to get stuck at locally fluctuations。
# maybe compare with the initial loss is best.
# local grad use std/mean
# the value of loss weight should be controled, if it is bigger, such as (1,10,100) and lr=0.01,
# which equals to (0.1,1,10) and lr=0.1, these will give rise to exploration of grad,and this will
# result in nan value in pred.
import torch
import numpy as np
from utils import print_log


# def f_m_sig(arr, max=100):
#     # mag_arr = 2 / (np.exp(-arr / arr.std()) + 1) - 1
#     mag_arr = 2 / (np.exp(-arr / abs(arr.max())) + 1) - 1
#     return max / (1 + np.power(max, mag_arr))
def f_m_sig(arr,max=300):

    a = 0.3  # a越小，weight比值拉的越大
    b = 10    # b越大，arr为小间隔时，比值也可拉大些，而不是约为1：1：1
    mag_arr = a/ (np.power(b,-arr/arr.std())+ a)
    return  (max+1)/(1+np.power(max,mag_arr/mag_arr.max()))


def get_grad(arr,  log=None, initial_loss=None, epoch=1):
    if isinstance(arr, (tuple, list)):
        arr = np.array(arr)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=-1)

    num = arr.shape[0]
    ep_num = num // epoch
    local_last_mean = arr[-ep_num:].mean(0)

    if initial_loss is not None:
        global_grad = (initial_loss - local_last_mean) / initial_loss / epoch
    else:
        global_grad = (arr[:ep_num].mean(0) - arr[-ep_num:].mean(0)) / arr[:ep_num].mean(0) / (epoch - 1)


    std_grad = arr.std(0) / arr.mean(0)  # 越大，代表变化越快

    if log is not None:
        log.info("now global_grad ,std_grad is : "
                 f"{(global_grad, std_grad)}\n")
    return  local_last_mean, std_grad*global_grad


class Grad_check():
    """Check invalid loss.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, model, warm_iter=100, grad_defend_iter=0, logger=None):
        self.warm_iter = warm_iter
        self.grad = dict()
        self.diff_grad = dict()
        self.max_min_grad = dict()
        self.grad_defend_iter = grad_defend_iter
        self.logger = logger
        self.model = model

    def after_train_iter(self, iter):
        if iter >= self.grad_defend_iter:
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:

                    if iter == self.grad_defend_iter:
                        self.grad[name] = [p.grad.mean() / p.data.mean(),
                                           p.grad.max() / p.data.mean(),
                                           p.grad.min() / p.data.mean()]
                        self.diff_grad[name] = [[] for _ in range(3)]
                        self.max_min_grad[name] = [0.] * 3
                        print_log(
                            f'in {name}, param (mean,max,min) is ({p.data.mean()},{p.data.max()},{p.data.min()}), '
                            f'grad (mean,max,min) is ({p.grad.mean()},{p.grad.max()},{p.grad.min()})', self.logger)
                    else:
                        defend_flag = 0
                        now_grad = [p.grad.mean() / p.data.mean(),
                                    p.grad.max() / p.data.mean(),
                                    p.grad.min() / p.data.mean()]
                        for i in range(3):
                            avg = self.grad[name][i]
                            ng = now_grad[i]
                            diff = ng / (avg + 1e-16)
                            if iter <= self.grad_defend_iter + self.warm_iter:
                                self.diff_grad[name][i].append(diff)
                                if iter == self.grad_defend_iter + self.warm_iter:
                                    self.max_min_grad[name][i] = (max(self.diff_grad[name][i]),
                                                                  min(self.diff_grad[name][i]))

                            else:
                                if diff > self.max_min_grad[name][i][0] or diff < min(self.max_min_grad[name][i][1],
                                                                                      -0.1):
                                    print_log(
                                        f'somethin eruption, in {name}, its {i}\n '
                                        f'grad average (mean,max,min) is {self.grad[name]}, '
                                        f'now grad (mean,max,min) is {now_grad}\n', self.logger)

                                    defend_flag = 1
                        if defend_flag:
                            p.grad = torch.zeros_like(p.grad)
                        else:
                            self.grad[name] = list(
                                map(lambda x, y: (x * iter + y) / (iter + 1), self.grad[name], now_grad)
                            )


class Loss_Auto_Weight_Hook():
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, model, loss, start_epoch=0, detect_epoch=2, grad_detect=False, adj_epoch=5, per_train=1,
                 loss_name=['box', 'cls', 'dfl'], loss_balance=[1., 1., 1.], logger=None):
        self.model = model
        self.detect_epoch = detect_epoch
        self.grad_detect = grad_detect
        self.loss_num = len(loss_name)
        self.loss_name = loss_name
        self.loss_weight = np.array([1.] * self.loss_num)
        # self.independent_local_grad = np.array([0.] * self.loss_num)
        self.independent_global_grad = np.array([0.] * self.loss_num)
        self.last_mean = None
        # self.initial_loss = np.array([0.] * self.loss_num)
        self.loss_ind = 0
        self.now_train = 0
        self.start_train = False
        self.adj_epoch = adj_epoch
        self.dictstate = None
        self.ema = 0.8 # min(adj_epoch / 10, 0.5)
        self.per_train = per_train
        # self.all_lw = []
        self.loss_balance = np.array(loss_balance)
        self.start_epoch = start_epoch
        self._epoch = start_epoch
        self.loss = loss
        self.logger = logger
        self.loss_contain = []
        self.print_loss_weight()
        # self.init_dict()

    def init_dict(self):
        self.grad = dict()
        self.diff_grad = dict()
        self.mean_std_grad = dict()
        self.loss_contain = []
        # print('now loss grad initialized')

    def get_state_dict(self):
        self.dictstate = dict()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # self.dictstate[name] = p.clone().detach()
                # self.dictstate[name] = p.new_tensor(p)
                self.dictstate[name] = p.clone()

    def reload_state_dict(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                with torch.no_grad():
                    p.copy_(self.dictstate[name])
                    p.requires_grad = True
    def set_loss_weight(self, loss_weight):
        for name, w in zip(self.loss_name, loss_weight):
            self.loss.hyp[name] = w

    def print_loss_weight(self):
        for name in self.loss_name:
            print_log(
                f"{name} loss weight is {self.loss.hyp[name]}", self.logger
            )
    def set_epoch(self):
        self._epoch += 1
        if self.detect_epoch != 0:
            if self._epoch % self.detect_epoch == 0 and not self.start_train:
                self._epoch = self.start_epoch

    def before_train_epoch(self, epoch):
        if self.detect_epoch == 0:
            self.init_dict()
            # self.set_loss_weight([1.,1.,1.])
        else:
            if epoch % self.detect_epoch == 0 and not self.start_train:
                self.init_dict()
                # self._epoch = self.start_epoch
                if self.dictstate is None:
                    self.get_state_dict()
                else:
                    self.reload_state_dict()
                    print_log('state dict reload', self.logger)
                print_log('now epoch and iter initialized', self.logger)
                # runner._inner_iter = 0
                if self.loss_ind > self.loss_num:  # and self.now_train > self.per_train
                    self.start_train = True

            if self.loss_ind <= self.loss_num:
                loss_weight = [0.1] * self.loss_num
                if self.loss_ind < self.loss_num:
                    loss_weight[self.loss_ind] = 3.
                else:
                    loss_weight = [1.] * self.loss_num
                    # self.independent_local_grad /= self.per_train
                    self.independent_global_grad /= self.per_train
                    # self.initial_loss /= self.per_train

                self.set_loss_weight(loss_weight)

            else:
                self.set_loss_weight(self.loss_weight)  # * self.loss_balance

        self.print_loss_weight()

    def after_train_iter(self, iter, loss_value):
        if self.loss_ind < self.loss_num:
            # self.loss_contain.append(runner.outputs['loss'].item())
            if self.grad_detect:
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        now_grad = [p.grad.mean() / p.data.mean(),
                                    p.grad.max() / p.data.mean(),
                                    p.grad.min() / p.data.mean()]
                        if name not in self.grad:
                            self.grad[name] = now_grad
                            self.diff_grad[name] = [[] for _ in range(3)]
                        else:
                            for i in range(3):
                                avg = self.grad[name][i]
                                ng = now_grad[i]
                                diff = ng / (avg + 1e-16)
                                self.diff_grad[name][i].append(diff.item())
                            self.grad[name] = list(
                                map(lambda x, y: (x * iter + y) / (iter + 1), self.grad[name], now_grad)
                            )
        # else:
            # for i in runner.outputs['log_vars'].values():
            #     print(i)
        iter_loss = loss_value.tolist()

        self.loss_contain.append(iter_loss)

    def after_train_epoch(self, epoch):
        if self.detect_epoch != 0:
            if self.loss_ind <= self.loss_num:
                if (epoch + 1) % self.detect_epoch == 0:
                    if self.loss_ind < self.loss_num:
                        if self.grad_detect:
                            layer_names = []
                            for name, p in self.model.named_parameters():
                                if p.requires_grad and p.grad is not None:
                                    self.mean_std_grad[name] = [[np.array(self.diff_grad[name][i]).mean(),
                                                                 np.array(self.diff_grad[name][i]).std()] for i in
                                                                range(3)]
                                    layer_names.append(name)
                            mean_std_grad_all = np.array(list(self.mean_std_grad.values()))  # (n,3,2)
                            max_index = np.argmax(mean_std_grad_all, axis=0)  # (3,2)
                            layer_count = np.bincount(max_index.flatten())
                            if layer_count.max() < 2:
                                layer_index = max_index[0][0]
                            else:
                                layer_index = np.argmax(layer_count)

                            print_log(
                                f"According to grad variation, {self.loss_name[self.loss_ind]} loss was detected huge influence in {layer_names[layer_index]}", self.logger
                            )

                        local_mean, global_grad = get_grad(self.loss_contain, log=self.logger,
                                                                           epoch=self.detect_epoch)

                        self.independent_global_grad[self.loss_ind] += global_grad[self.loss_ind]
                        # self.independent_local_grad[self.loss_ind] += local_std_grad[0]
                        # self.initial_loss[self.loss_ind] += local_mean

                        self.now_train += 1
                        if self.now_train == self.per_train:
                            self.now_train = 0
                            self.loss_ind += 1

                    else:
                        local_mean, global_grad = get_grad(self.loss_contain, epoch=self.detect_epoch,
                                                                           log=self.logger)

                        # local_weight = f_m_sig(local_std_grad / self.independent_local_grad)
                        #
                        # runner.logger.info(
                        #     f'locally, the independent loss grad is {self.independent_local_grad}\n and dependent grad is: {local_std_grad},now loss_weight is {local_weight}\n')
                        #
                        print_log(f'globally, the independent loss grad is {self.independent_global_grad}\n and dependent grad is: {global_grad}', self.logger)
                        # self.independent_global_grad = np.where(global_grad < self.independent_global_grad, self.independent_global_grad, global_grad)

                        global_weight = f_m_sig(global_grad / self.independent_global_grad)
                        # loss_weight = self.ema * loss_weight + (1 - self.ema )  * self.independent_local_grad

                        loss_weight = global_weight/(global_weight.sum()/3)

                        self.loss_weight = loss_weight*100//1/100 * self.loss_balance

                        # loss_weight /= (loss_weight.sum(0) / loss_weight.shape[0])

                        print_log(
                            f'now, the global_weight is {global_weight}\n ,now loss_weight is {self.loss_weight}\n', self.logger)
                        # max_id = np.where(loss_weight>loss_weight.sum() * 0.5)[0]  #若无，返回【】
                        # self.all_lw.append((loss_weight,max_id))


                        self.loss_ind += 1

            else:
                if (epoch + 1) % self.adj_epoch == 0:
                    local_mean, global_grad = get_grad(self.loss_contain, epoch=self.adj_epoch,
                                                                       log=self.logger, initial_loss=self.last_mean)

                    # local_weight = f_m_sig(local_std_grad / self.independent_local_grad)
                    # runner.logger.info(
                    #     f'locally, the independent loss grad is {self.independent_local_grad}\n and dependent grad is: {local_std_grad},now loss_weight is {local_weight}\n')
                    # global_grad /= (runner.epoch - (
                    #             self.adj_epoch + self.detect_epoch) / 2)  # loss曲线逐渐平缓，global grad 必然越来越小。
                    global_weight = f_m_sig(global_grad / self.independent_global_grad)
                    print_log(
                        f'globally, the independent loss grad is {self.independent_global_grad}\n and dependent grad is: {global_grad},now loss_weight is {global_weight}\n', self.logger)

                    loss_weight = global_weight / (global_weight.sum() / 3)

                    loss_weight *= 100 // 1 / 100 * self.loss_balance

                    # loss_weight /= (loss_weight.sum(0) / loss_weight.shape[0]) * self.loss_balance
                    print_log(f'the loss_weight before is {self.loss_weight}\n and now combined with: {loss_weight}\n', self.logger)
                    # self.loss_weight = self.ema * loss_weight + runner.model.module.bbox_head.loss_weight * (
                    #         1 - self.ema)
                    self.loss_weight = (self.ema * loss_weight +  (1 - self.ema) * self.loss_weight)*100//1/100
                    # self.loss_weight /= (self.loss_weight.sum(0) / self.loss_weight.shape[0])
                    self.loss_contain = []
                    self.last_mean = local_mean









