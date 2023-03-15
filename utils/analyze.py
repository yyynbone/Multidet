# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy
import re
# from mmdet.core.hook.loss_grad_hook import get_grad_next

def smooth(data, sm=3):
    if sm>1:
        k = np.ones(sm)*1./sm
        data = np.convolve(k, data, mode='same')
    #print(data[-2], data[-1])
    data[-1] = data[-2]
    return data
def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, iter_plot, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is not None:
        org_legend = [le.replace("_", " ") for le in legend]
    marker = ['o', 'v', 'D', 's', 'd', '^', '*', '+']
    plt.figure(figsize=(12,9))   #plus 100
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        new_metric = []
        for metric in args.keys:
            for ep in epochs:
                for key in log_dict[ep]:
                    if metric in key:
                        new_metric.append(key)
        metrics = set(new_metric)
        num_metrics = len(metrics)
        if not num_metrics:
            continue
        if legend is None:
            legend = []
            for l, json_log in enumerate(args.json_logs):
                # file_name = json_log.split('/')[-1].replace(".log.json","")
                file_name = json_log.split('/')[-1].split('.')[-3]
                # if 'loss' in file_name:
                #     number = re.findall('\d+',file_name)
                #     if number:
                #         c, o, b = list(map(lambda x:int(x)/100, number))
                #         file_name = f'{c}:{o}:{b} loss weight '
                #     else:
                #         file_name = file_name.replace("_"," ")
                for metric in metrics:
                    legend.append(f'{metric} for {file_name}')
                    # legend.append(f'{metric}')
        # assert len(legend) == (len(args.json_logs) * len(args.keys))
        else:

            if num_metrics>1:
                new_le = []
                for l, json_log in enumerate(args.json_logs):
                    for metric in metrics:
                        new_le.append(f'{metric} for {org_legend[l]}')
                legend = new_le
            else:
                legend = org_legend
            plt.title("mAP@.5:.95 based on different loss weight")



        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            # if metric not in log_dict[epochs[0]]:
            #     raise KeyError(
            #         f'{args.json_logs[i]} does not contain metric {metric}')

            if 'AP' in metric or 'AR' in metric:
                # xs = np.arange(1, max(epochs) + 1)
                xs = []
                ys = []
                for epoch in epochs:
                    if metric in log_dict[epoch].keys():
                        xs.append(epoch)
                        ys += log_dict[epoch][metric]
                ax = plt.gca()

                ys = smooth(ys)
                # ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker=marker[i], markersize=3)
                plt.legend()
            else:
                if iter_plot:
                    xs = []
                    ys = []
                    vxs = []
                    vys = []
                    for epoch in epochs:
                        iters = log_dict[epoch]['iter']
                        if log_dict[epoch]['mode'][-1] == 'val':
                            iters = iters[:-1]
                        num_iters_per_epoch = iters[-1]
                        xs.append(
                            np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                        ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                    xs = np.concatenate(xs)
                    ys = np.concatenate(ys)
                    xlabel = 'iter'

                else:
                    xs = []
                    ys = []
                    vxs = []
                    vys = []
                    for epoch in epochs:
                        if metric in log_dict[epoch].keys():
                            train_count = log_dict[epoch]['mode'].count('train')
                            val_count = log_dict[epoch]['mode'].count('val')
                            if val_count > 1:
                                vxs.append(epoch)
                                vys.append(np.array(log_dict[epoch][metric][train_count:]).mean())
                            if train_count > 1:
                                ys.append(np.array(log_dict[epoch][metric][:train_count]).mean())
                                xs.append(epoch)
                    xlabel = 'epoch'
                if sum(ys):
                    plt.xlabel(xlabel)
                    plt.plot(
                        xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
                    if vys:
                        plt.plot(
                            vxs, vys, label=legend[i * num_metrics + j]+'_val', marker='o', linewidth=1.)
                    plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def plot_wmcurve(log_dicts, iter_plot, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend

    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        new_metric = []
        for metric in args.keys:
            for ep in epochs:
                for key in log_dict[ep]:
                    if metric in key:
                        new_metric.append(key)
        metrics = set(new_metric)
        if legend is None:
            legend = []
            for i, json_log in enumerate(args.json_logs):
                for metric in metrics:
                    legend.append(f'{i}_{metric}')
                    # legend.append(f'{metric}')
        # assert len(legend) == (len(args.json_logs) * len(args.keys))
        num_metrics = len(metrics)

        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            # if metric not in log_dict[epochs[0]]:
            #     raise KeyError(
            #         f'{args.json_logs[i]} does not contain metric {metric}')

            if 'AP' in metric or 'AR' in metric:
                # xs = np.arange(1, max(epochs) + 1)
                xs = []
                ys = []
                for epoch in epochs:
                    if metric in log_dict[epoch].keys():
                        xs.append(epoch)
                        ys += log_dict[epoch][metric]
                ax = plt.gca()
                # ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
                plt.legend()
            else:
                if iter_plot:
                    xs = []
                    ys = []
                    num_iters_per_epoch = log_dict[epochs[0]]['iter'][-2]
                    for epoch in epochs:
                        iters = log_dict[epoch]['iter']
                        if log_dict[epoch]['mode'][-1] == 'val':
                            iters = iters[:-1]
                        xs.append(
                            np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                        ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                    xs = np.concatenate(xs)
                    ys = np.concatenate(ys)
                    plt.xlabel('iter')
                    plt.plot(
                        xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
                    plt.legend()
                else:
                    xs = []
                    ys = []

                    vxs = []
                    vys = []
                    for epoch in epochs:
                        if metric in log_dict[epoch].keys():
                            train_count = log_dict[epoch]['mode'].count('train')
                            val_count = log_dict[epoch]['mode'].count('val')
                            if val_count > 1:
                                vxs.append(epoch)
                                vys.append(np.array(log_dict[epoch][metric][train_count:]).mean())
                            if train_count > 1:
                                ys.append(np.array(log_dict[epoch][metric][:train_count]).mean())
                                xs.append(epoch)
                plt.xlabel('epoch')
                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
                if vys:
                    plt.plot(
                        vxs, vys, label=legend[i * num_metrics + j]+'_val', marker='o', linewidth=1.)
                plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()

def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args



def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                ##################################
                # new for none val
                # if 'val'== log['mode'] and len(list(log.keys()))==4:
                if len(list(log.keys()))< 5 :
                    print("pop: ",log )
                    continue
                ######################################
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts

def get_iter_idx(val_dict):
    # except val mode
    val_idx = []
    train_v = []

    for i, k in enumerate(val_dict['iter']):
        if val_dict['mode'][i] == 'val':
            val_idx.append(i)
        else:
            train_v.append(k)

    # # filter_index =  v.index(v[0], -1)
    # filter_count = train_v.count(train_v[0])
    #
    # # num, last = divmod( len(v), filter_count)
    # if len(train_v) % filter_count:
    #     filter_count -= 1
    # return filter_count

    last_idx =  train_v[::-1].index(train_v[0])
    return last_idx

def get_count(val_dict):
    # except val mode
    train_v = []

    for i, k in enumerate(val_dict['iter']):
        if val_dict['mode'][i] == 'train':
            train_v.append(k)

    # filter_index =  v.index(v[0], -1)
    filter_count = train_v.count(train_v[0])

    # num, last = divmod( len(v), filter_count)
    if len(train_v) % filter_count:
        filter_count -= 1
    return filter_count


# def get_iter_idx(val_dict):
#     # except val count
#     val_count = val_dict['mode'].count('val')
#     v = val_dict['iter']
#
#     # filter_index =  v.index(v[0], -1)
#     filter_count = v.count(v[0]) - val_count
#
#     # num, last = divmod( len(v), filter_count)
#     if len(v) % filter_count:
#         filter_count -= 1
#     return filter_count

def filter(o_log_dicts):
    log_dicts = deepcopy(o_log_dicts)
    warm_dicts = []
    for log_dict in log_dicts:
        warm_log_dict = dict()
        for epoch,val_dict in log_dict.items():
            warm_dict = dict()
            last_idx = get_iter_idx(val_dict)
            iter_l = len(val_dict['iter'])
            val_get = False
            if val_dict['mode'][-1] == 'val':
                val_get = True
            for k,v in val_dict.items():
                if len(v)<iter_l:
                    if val_get:
                        warm_dict[k] = v[:-1]
                        val_dict[k] = v[-1:]
                    else:
                        warm_dict[k] = v
                        val_dict[k] = []
                else:
                    warm_dict[k] = v[:-last_idx]
                    val_dict[k] = v[-last_idx:]

            if list(warm_dict.keys()):
                warm_log_dict[epoch] = warm_dict
        if list(warm_log_dict.keys()):
            warm_dicts.append(warm_log_dict)
    return log_dicts,warm_dicts


def warm_get(log_dicts):
    warm_dicts = []
    for log_dict in log_dicts:

        filter_count = get_count(list(log_dict.values())[0])

        i=0
        while i < filter_count:
            warm_log_dict = dict()
            for epoch, val_dict in log_dict.items():
                warm_dict = dict()
                for k,v in val_dict.items():

                    num, last = divmod(len(v), filter_count)

                    warm_dict[k] = v[i*num:(i+1)*num]
                warm_log_dict[epoch] = warm_dict
            i+=1
            warm_dicts.append(warm_log_dict)
    return warm_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    f_log_dicts,warm_dicts = filter(log_dicts)

    eval(args.task)(f_log_dicts, False, args)
    # if warm_dicts:
    #     warm_dicts = warm_get(warm_dicts)
    #     for warm_dict in warm_dicts:
    #         eval(args.task)([warm_dict], True, args)


if __name__ == '__main__':
    main()
