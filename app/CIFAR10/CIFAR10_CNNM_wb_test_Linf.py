#%%
import os
import sys
sys.path.append('../../core')
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import time
from CIFAR10_Dataset_new import get_dataloader, get_dataloader_bba
from Evaluate import test, test_rand, cal_AUC_robustness
from Evaluate_advertorch import test_adv, test_adv_auto
from Evaluate_bba_spsa import test_adv as test_adv_spsa
from advertorch_examples.models import get_cifar10_wrn28_widen_factor
#%%
def Net(net_name):
    if net_name == 'mmacifar10':
        
        model = get_cifar10_wrn28_widen_factor(4)
    elif net_name == 'mmacifar10g':
        from models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(4)
    elif net_name == 'wrn28_10':
        from advertorch_examples.models import get_cifar10_wrn28_widen_factor
        model = get_cifar10_wrn28_widen_factor(10)
    else:
        raise ValueError('invalid net_name ', net_name)
    return model

#%%
def main():
    net_name = "mmacifar10"
    device= device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
    norm_type=np.inf
    noise_norm_list = [2/255, 4/255, 6/255,8/255]
    #noise_norm_list = [6/255]
    model=get_cifar10_wrn28_widen_factor(4)
    #noise_norm_list = [0.5]
    #loader_bba = get_dataloader_bba()
    loader_train,loader_test = get_dataloader()
    del loader_train
    base = "results_for_test/"

    
    tasks = ["EE16","EE8","GAI8","GAI16","MMA12","MMA20","AME","FAT8","FAT16",
             "TRADES1_8","TRADES1_16","Madry4","Madry8","TE8","TE12"]
    modelToTests = []
    for task in tasks:
        modelToTests.extend([name.split(".pt")[0] for name in os.listdir(base) if task in name and "test" not in name])
    assert len(modelToTests) == len(tasks), "model not found"
    print ("all models found!")
    for i, model_name in enumerate(modelToTests):
        main_evaluate(base+model_name, tasks[i], model, device, 'test', loader_test, norm_type, noise_norm_list)

#%%
def main_evaluate(filename, task, model, device, data_name, loader, norm_type, noise_norm_list):
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))    
    if "MMA" in task:
        model.load_state_dict(checkpoint['model'])
    elif "TE" in task:
        model.load_state_dict(checkpoint)
    elif "LBGAT" in task:
        model = nn.DataParallel(model).to(device)
        model.load_state_dict(checkpoint)
    elif "EE" in task or "AWP" in task:
        model.load_state_dict(checkpoint['state_dict'])
    elif "LAS" in task:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    result_100pgd=[]
    result_20pgd=[]
    result_100if=[]
    result_20if=[]
    result_fgsm=[]
    result_auto=[]
    result_cwpgd=[]

    #%% white uniform noise attack
    """
    for noise_norm in [8/255, 16/255, 32/255]:
        result_uniform.append(test_rand(model, device, loader, 10, noise_norm=noise_norm, norm_type = norm_type, max_iter=100))
    noise=[0]
    acc=[result_uniform[0]['acc_clean']]
    for k in range(0, len(result_uniform)):
        noise.append(result_uniform[k]['noise_norm'])
        acc.append(result_uniform[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('uniform auc is ', auc)  
    """
    
    #%% fgsm
    """
    num_repeats=1  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_fgsm.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=1, step=noise_norm, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_fgsm[0]['acc_clean']]
    for k in range(0, len(result_fgsm)):
        noise.append(result_fgsm[k]['noise_norm'])
        acc.append(result_fgsm[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('fgsm auc is ', auc)
    
    #%% ifgsm20
    
    num_repeats=1  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_20if.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=20, step=noise_norm/4, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_20if[0]['acc_clean']]
    for k in range(0, len(result_20if)):
        noise.append(result_20if[k]['noise_norm'])
        acc.append(result_20if[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('if20 auc is ', auc)
    
    #%% ifgsm100
    num_repeats=1  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_100if.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_100if[0]['acc_clean']]
    for k in range(0, len(result_100if)):
        noise.append(result_100if[k]['noise_norm'])
        acc.append(result_100if[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('if100 auc is ', auc)
    
    #%% 20pgd
    num_repeats=1  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_20pgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=20, step=noise_norm/4, method='pgd', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_20pgd[0]['acc_clean']]
    for k in range(0, len(result_20pgd)):
        noise.append(result_20pgd[k]['noise_norm'])
        acc.append(result_20pgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('pgd20 auc is ', auc)
    #%% 100pgd
    num_repeats=1  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_100pgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='pgd', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('pgd100 auc is ', auc)
#%%pgd cw
    num_repeats=2  
    for noise_norm in noise_norm_list:
        start = time.time()
        result_cwpgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='pgd_ce_cw', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_cwpgd[0]['acc_clean']]
    for k in range(0, len(result_cwpgd)):
        noise.append(result_cwpgd[k]['noise_norm'])
        acc.append(result_cwpgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('pgdcw auc is ', auc) 
    """
    #%% auto attack   
    num_repeats=1   
    for noise_norm in noise_norm_list:
        start = time.time()
        result_auto.append(test_adv_auto(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='auto', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_auto[0]['acc_clean']]
    for k in range(0, len(result_auto)):
        noise.append(result_auto[k]['noise_norm'])
        acc.append(result_auto[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('auto auc is ', auc) 
    #%%  
    savename= filename +'_result_non_auto_wba_L'+str(norm_type)+'_r'+str(num_repeats)+'_'+data_name+'.pt'
    torch.save({'result_100pgd':result_100pgd, 
                'result_auto':result_auto,
                'result_cwpgd':result_cwpgd,
                'result_20pgd':result_20pgd,
                'result_100if':result_100if,
                'result_20if':result_20if,
                'result_fgsm':result_fgsm}, savename)
    print('saved:', savename)

#%%
if __name__ == "__main__":
    #main_evaluate()
    model = 0
    device = 0
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(device)
    main()