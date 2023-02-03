import os
import argparse
import numpy as np
import matplotlib.pyplot as plt # 画图工具
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import copy

def get_results(args, return_mat=False):
    result_path = os.path.join("./results", args.dataset)
    # result_recording_path = f'{result_path}/{args.method}_{args.cssl}_{args.sum_cmt}.txt'

    eval_post = '_eval' if args.eval else ''

    avg_acc_list = []
    avg_frg_list = []
    mean_result = []
    for i in range(1,5):
        path_exist = False
        if args.sum_cmt == '':
            temp_path = f'{result_path}/{args.method}_{args.cssl}_{i}.txt'
            path_exist = path_exist or os.path.exists(temp_path)
        else:
            for note in ['', '-', '_']:
                temp_path = f'{result_path}/{args.method}_{args.cssl}_{args.sum_cmt}{note}{i}{eval_post}.txt' 
                # print(temp_path)

                path_exist = path_exist or os.path.exists(temp_path)
                if os.path.exists(temp_path):
                    break
        
        if path_exist:
            result_recording_path = temp_path
        else:
            continue

        with open(result_recording_path, 'r') as f:
            lines = f.readlines()
            avg_acc_idx, start_idx = 0, 0
            for i, line in enumerate(lines):
                # print(line)
                if line.split() == []:
                    continue
                if 'Accuracy matrix' in line:
                    avg_acc_idx = i - 1
                    start_idx = i + 1

                    break
            
            avg_acc = lines[avg_acc_idx].split()[1:]
            data_length = len(avg_acc)
            result_mat = lines[start_idx: start_idx + data_length]

            avg_acc = np.loadtxt(avg_acc)
            result = np.loadtxt(result_mat)
            mean_result.append(result)
            
            avg_frg = np.zeros(data_length-1)
            for j in range(1, data_length):
                sub_result = result[:j+1, :j+1]
                max_acc = sub_result.max(1)

                avg_frg[j-1] = (sub_result[:, -1] - max_acc).mean()
            
            avg_acc_list.append(avg_acc)
            avg_frg_list.append(avg_frg)
    
    if return_mat:
        return np.array(mean_result)
    else:
        avg_acc_list = np.array(avg_acc_list)
        avg_frg_list = np.array(avg_frg_list)

        return avg_acc_list, avg_frg_list

def get_avg_std(acc, frg):
    print(acc[:, -1].shape)
    
    acc_str = f'{np.around(acc[:, -1].mean(),2)}±{np.around(acc[:, -1].std(),2)}'
    frg_str = f'{-np.around(frg[:, -1].mean(),2)}±{np.around(frg[:, -1].std(),2)}'

    print(acc_str)
    print(frg_str)

def plot_learning_fig(args, methods, cmts):
    # frg_dict = {}
    mult_acc = [86.31, 85.32]
    if args.plot_frg:
        post_fix = '_frg'
    else:
        post_fix = ''
    print(post_fix)
    official_name = {'finetune': 'Finetune', 'mixup': 'LUMP', 'cassle': 'CaSSLe', 'cassle_uniform': 'Ours'}  
    for j, settings in enumerate([20, 10]):
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        idx = list(range(settings))
        # acc_dict = {}

        print("setting num:", settings)
        for i, method in enumerate(methods):
            args.method = method
            args.sum_cmt = cmts[j][i]

            print(args.method)

            if args.plot_frg:
                # print(f"figures/DataSetting_5{post_fix}.pdf")
                _, avg_frg_list = get_results(args)            
                avg_frg, std_frg = avg_frg_list.mean(0), avg_frg_list.std(0)
                avg_frg = -np.insert(avg_frg, 0, 0)
                std_frg = -np.insert(std_frg, 0, 0)
                # print(avg_frg)
                axs.errorbar(x=idx, y=avg_frg, yerr=std_frg, label=official_name[method], linewidth=2, capsize=4)
            else:
                avg_acc_list, _ = get_results(args)            
                avg_acc, std_acc = avg_acc_list.mean(0), avg_acc_list.std(0)
                axs.errorbar(x=idx, y=avg_acc, yerr=std_acc, label=official_name[method], linewidth=2, capsize=4)

            # avg_frg, std_frg = avg_frg_list.mean(0), avg_frg_list.mean(0)
            # acc_dict[method] = [avg_acc, std_acc]
        
        axs.set_xlabel('Number of data sets learnt')
        axs.legend()
        # axs[j].hlines(mult_acc[j], xmin=0, xmax=settings-1, label='Multitask', linewidth=2)
    # axs.errorbar(x=idx, y=pca_acc, yerr=pca_std, label='Max Entropy', color="#c23728", ecolor="#e1a692", linewidth=2, capsize=4)
        if args.plot_frg:
            axs.set_ylabel('Average Forgetting')
        else:
            axs.set_ylabel('Average Accuracy')
        if j == 0:
            axs.set_xticks([0, 4, 9, 14, 19], ['1', '5', '10', '15', '20'])
        else:
            axs.set_xticks(idx, [str(k+1) for k in idx])
    
        # axs[j].set_title(f"{int(100 / settings)} classes per subset")
    # fig.legend()
        fig.tight_layout()
        if j == 0:
            # plt.savefig(f"figures/DataSetting_5{post_fix}.pdf")
            plt.savefig(f"figures/DataSetting_5.pdf")
        else:
            # plt.savefig(f"figures/DataSetting_10{post_fix}.pdf")
            plt.savefig(f"figures/DataSetting_10.pdf")
    # plt.show()


def get_time(args):
    
    result_path = os.path.join("./results", args.dataset)
    # result_recording_path = f'{result_path}/{args.method}_{args.cssl}_{args.sum_cmt}.txt'

    eval_post = '_eval' if args.eval else ''

    time_list = []
    for i in [1, 2, 3, 4]:
        path_exist = False
        if args.sum_cmt == '':
            temp_path = f'{result_path}/{args.method}_{args.cssl}_{i}.txt'
            print(temp_path)
            path_exist = path_exist or os.path.exists(temp_path)
        else:
            for note in ['', '_', '-']:
                temp_path = f'{result_path}/{args.method}_{args.cssl}_{args.sum_cmt}{note}{i}{eval_post}.txt' 
                # print(temp_path)

                path_exist = path_exist or os.path.exists(temp_path)
                if os.path.exists(temp_path):
                    break
        
        if path_exist:
            print(temp_path)
            result_recording_path = temp_path
        else:
            continue

        with open(result_recording_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'total time' in line:
                    time_list.append(int(line.split()[-1]))
                    break
    
    mean_time = np.round(np.array(time_list).mean() / 3600, 2)
    std_time = np.round((np.array(time_list) / 3600).std(), 2)
    return mean_time, std_time

def plot_frg_mat(args, methods, cmts):
    # frg_dict = {}
    # mult_acc = [86.31, 85.32]
    official_name = {'finetune': 'Finetune', 'mixup': 'LUMP', 'cassle': 'CaSSLe', 'cassle_uniform': 'Ours', 'si': 'SI', 'der':'DER'}
    dataset_name = {'seq-cifar10': 'CIFAR-10', 'seq-cifar100': 'CIFAR-100'}
    for j, dataset in enumerate(['seq-cifar10', 'seq-cifar100']):
        fig, axs = plt.subplots(1, 6, figsize=(18, 3))
        fig.subplots_adjust(wspace=0.2)
        args.dataset = dataset
        if dataset == 'seq-cifar10':
            length = 5
        else:
            length = 20

        temp_results = []
        for i, method in enumerate(methods):
            args.method = method
            args.sum_cmt = cmts[j][i]

            print(args.method)

            result = get_results(args, return_mat=True)            
            result = result.mean(0)

            # print(result)
            temp_result = copy.deepcopy(result)
            for k in range(0, length):
                max_acc = result[:, :k+1].max(1)
                temp_result[:, k] = (max_acc - result[:, k])
            
            temp_results.append(temp_result)
        
        temp_results = np.array(temp_results)

        temp_results = np.log(temp_results + 1) / 2
        # max_calc = temp_results.flat[temp_results.flat > 0]
        # print(np.histogram(max_calc))
        # max, min = np.sort(temp_results.flat)[-int(4 / 15 * len(temp_results.flat))], temp_results.min()
        max, min = temp_results.max(), temp_results.min()
        
        def add_right_cas(ax, pad, width):
            axpos = ax.get_position()
            caxpos = mpl.transforms.Bbox.from_extents(
                axpos.x1 + pad,
                axpos.y0,
                axpos.x1 + pad + width,
                axpos.y1
            )
            cax = ax.figure.add_axes(caxpos)

            return cax

        print('max:', max)
        print('min:', min)
        for i, method in enumerate(methods):
            x = np.arange(1, length+1)
            y = np.arange(1, length+1)
            
            if i == len(methods) - 1:
                im = axs[i].pcolormesh(x, y, temp_results[i], vmin=min, vmax=max, edgecolors='none', snap=True, cmap='Reds')
                cax = add_right_cas(axs[i], pad=0.01, width=0.01)
            else:
                axs[i].pcolormesh(x, y, temp_results[i], vmin=min, vmax=max, edgecolors='none', snap=True, cmap='Reds')
            axs[i].set_title(f"{official_name[method]}", fontdict={'fontsize': 15})
            axs[i].set_aspect('equal')
            idx = list(range(length))
            if length == 5:
                axs[i].set_yticks([k+1 for k in idx], [str(k+1) for k in idx])
                axs[i].set_xticks([k+1 for k in idx], [str(k+1) for k in idx])
            else:
                axs[i].set_yticks([1, 5, 10, 15, 20], ['1', '5', '10', '15', '20'])
                axs[i].set_xticks([1, 5, 10, 15, 20], ['1', '5', '10', '15', '20'])
            
        color_bar = fig.colorbar(im, cax=cax)
        # fig.set_layout_engine('tight')
        # plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.1)
        plt.savefig(f"figures/Forgetting_{dataset_name[dataset]}.pdf")
    # plt.show()

def plot_acc_time(args):
    official_name = {'finetune': 'Finetune', 'mixup': 'LUMP', 'cassle': 'CaSSLe', 'cassle_uniform': 'Ours', 'si': 'SI', 'der':'DER'}
    methods = ['finetune', 'si', 'der', 'mixup', 'cassle', 'cassle_uniform']
    cmts = [['seed', 'seed', 'seed', 'seed', 'seed', 'entropy_buf32_seed'],['seed', 'seed', 'buf640_seed', 'buf640_seed', 'seed', 'noise_10nei_seed']]
    for j in range(1, 2):
        time_avg_list, time_std_list = [], []
        # acc_avg_list, acc_std_list = [], []
        fig, axs = plt.subplots(1, 1, figsize=(3, 3))
        for method, cmt in zip(methods, cmts[j]):
            args.method = method
            args.cmt = cmt
            
            mean_time, std_time = get_time(args)
            
            time_avg_list.append(mean_time)
            time_std_list.append(std_time)

            acc_list, _ = get_results(args)
            
            # acc_avg_list.append(acc_list.mean(0)[-1])
            # acc_std_list.append(acc_list.std(0)[-1])        

            axs.scatter(mean_time, acc_list.mean(0)[-1])
            axs.errorbar(mean_time, acc_list.mean(0)[-1], xerr=std_time, yerr=acc_list.std(0)[-1], label=official_name[method], linewidth=2, capsize=4)
            # mean_acc = f'{np.around(acc[:, -1].mean(),2)}±{np.around(acc[:, -1].std(),2)}'
        
        fig.tight_layout()

        if j == 0:
            plt.savefig('figures/Time_vs_Acc_C10')
        else:
            plt.savefig('figures/Time_vs_Acc_C100')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-type', type=str, default='avg_std')
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-method', type=str, default=None)
    parser.add_argument('-cssl', type=str, default=None)
    parser.add_argument('-sum_cmt', type=str, default='')
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-plot_frg',  action='store_true')
    args = parser.parse_args()

    if args.type == 'avg_std':
        avg_acc, avg_frg = get_results(args)
        get_avg_std(avg_acc, avg_frg)
    elif args.type == 'time':
        methods = ['finetune', 'si', 'der', 'mixup', 'cassle', 'cassle_uniform']
        cmts = [['seed', 'seed', 'seed', 'seed', 'seed', 'entropy_buf32_seed'],['seed', 'seed', 'buf640_seed', 'buf640_seed', 'seed', 'noise_10nei_seed']]
        mean_time, std_time = get_time(args)
        print(f'{mean_time}±{std_time}')
        # print(std_time)
    elif args.type == 'plot':
        methods = ['finetune', 'mixup', 'cassle', 'cassle_uniform']
        cmts = [['seed', 'buf640_seed', 'seed', 'noise_10nei_seed'], ['10class_seed','10class_buf320_300epoch_seed','10class_300epc_seed','10class_300epc_seed']]
        plot_learning_fig(args, methods, cmts)
    elif args.type == 'frg_mat':
        methods = ['finetune', 'si', 'der', 'mixup', 'cassle', 'cassle_uniform']
        cmts = [['seed', 'seed', 'seed', 'seed', 'seed', 'entropy_buf32_seed'],['seed', 'seed', 'buf640_seed', 'buf640_seed', 'seed', 'noise_10nei_seed']]
        plot_frg_mat(args, methods, cmts)
    elif args.type == 'plot_time':
        plot_acc_time(args)