import os
import argparse
import numpy as np
import matplotlib.pyplot as plt # 画图工具


def get_results(args):
    result_path = os.path.join("./results", args.dataset)
    # result_recording_path = f'{result_path}/{args.method}_{args.cssl}_{args.sum_cmt}.txt'

    eval_post = '_eval' if args.eval else ''

    avg_acc_list = []
    avg_frg_list = []
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
            
            avg_frg = np.zeros(data_length-1)
            for j in range(1, data_length):
                sub_result = result[:j+1, :j+1]
                max_acc = sub_result.max(1)

                avg_frg[j-1] = (sub_result[:, -1] - max_acc).mean()
            
            avg_acc_list.append(avg_acc)
            avg_frg_list.append(avg_frg)
        
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
    official_name = {'finetune': 'finetune', 'mixup': 'LUMP', 'cassle': 'CaSSLe', 'cassle_uniform': 'Ours'}
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    
    for j, settings in enumerate([20, 10]):
        idx = list(range(settings))
        acc_dict = {}

        print("setting num:", settings)
        for i, method in enumerate(methods):
            args.method = method
            args.sum_cmt = cmts[j][i]

            print(args.method)

            avg_acc_list, avg_frg_list = get_results(args)
            
            avg_acc, std_acc = avg_acc_list.mean(0), avg_acc_list.std(0)
            axs[j].errorbar(x=idx, y=avg_acc, yerr=std_acc, label=official_name[method], linewidth=2, capsize=4)

            # avg_frg, std_frg = avg_frg_list.mean(0), avg_frg_list.mean(0)
            acc_dict[method] = [avg_acc, std_acc]
        
        axs[j].set_xlabel('Number of data sets learnt')
        axs[j].legend()
        # axs[j].hlines(mult_acc[j], xmin=0, xmax=settings-1, label='Multitask', linewidth=2)
    # axs.errorbar(x=idx, y=pca_acc, yerr=pca_std, label='Max Entropy', color="#c23728", ecolor="#e1a692", linewidth=2, capsize=4)
    
        axs[j].set_ylabel('Average Accuracy')
    # axs.set_xticks(idx, ['0', '320', '640', '1280'])
    
        axs[j].set_title(f"{int(100 / settings)} classes per subset")
    # fig.legend()
    fig.tight_layout()
    plt.savefig("figures/DataSetting.pdf")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-method', type=str, default=None)
    parser.add_argument('-cssl', type=str, default=None)
    parser.add_argument('-sum_cmt', type=str, default='')
    parser.add_argument('-eval', action='store_true')
    args = parser.parse_args()

    # avg_acc, avg_frg = get_results(args)
    # get_avg_std(avg_acc, avg_frg)
    methods = ['finetune', 'mixup', 'cassle', 'cassle_uniform']
    cmts = [['seed', 'buf640_seed', 'seed', 'noise_10nei_seed'], ['10class_seed','10class_buf320_300epoch_seed','10class_300epc_seed','10class_300epc_seed']]
    plot_learning_fig(args, methods, cmts)