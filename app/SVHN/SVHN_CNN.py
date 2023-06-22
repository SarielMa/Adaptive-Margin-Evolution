import sys
sys.path.append('../../core')
#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import os
from Evaluate import test, test_rand, cal_AUC_robustness#, test_adv
from Evaluate_advertorch import test_adv, test_adv_auto
from SVHN_Dataset import get_dataloader
from torch.optim import lr_scheduler
#%%

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



def Net(net_name):
    if net_name == 'Net2':
        print ("resnet18 in fact")
        return ResNet18()
    else:
        raise ValueError('unknown Net')

#%%
def save_checkpoint(filename, model, optimizer, result, epoch):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    #-------------------------------------------
    torch.save({'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def plot_result(loss_train_list, acc_train_list,
                acc_val_list, acc_other_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='train loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='train acc')
    ax[1].plot(acc_val_list, '-r', label='val acc')
    ax[1].set_xlabel('epoch')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(acc_other_list, '-m', label='other acc')
    ax[2].set_xlabel('epoch')
    ax[2].legend()
    ax[2].grid(True)
    return fig, ax
#%%
def get_filename(net_name, loss_name, epoch=None, pre_fix='result/SVHN_CNNM_'):
    if epoch is None:
        filename=pre_fix+net_name+'_'+loss_name
    else:
        filename=pre_fix+net_name+'_'+loss_name+'_epoch'+str(epoch)
    return filename
#%%
def main(epoch_start, epoch_end, train, arg, evaluate_model):
    main_train(epoch_start, epoch_end, train, arg)
    if evaluate_model == True:
        main_evaluate(epoch_end-1, arg)
#%%
def main_evaluate(epoch, arg):
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    device=arg['device']
    norm_type=arg['norm_type']
    #loader_bba = get_dataloader_bba()
    loader_train, loader_val, loader_test = get_dataloader(batch_size=256)
    #del loader_train, loader_val
    #main_evaluate_rand(net_name, loss_name, epoch, device, loader_test, noise_level_list)
    if norm_type == np.inf:
        #noise_norm_list=(1/255, 2/255, 4/255, 6/255, 8/255, 10/255)
        #noise_norm_list=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15)
        noise_norm_list=(2/255, 4/255, 6/255, 8/255)
        #noise_norm_list=(0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
        print('Linf norm noise_norm_list', noise_norm_list)
    else:
        #noise_norm_list=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        #noise_norm_list=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
        #noise_norm_list=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
        #noise_norm_list=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        #noise_norm_list=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
        noise_norm_list=(0.3,0.6,0.9)
        print('L2 norm noise_norm_list', noise_norm_list)
    #main_evaluate_bba_spsa(net_name, loss_name, epoch, device, loader_bba, norm_type, noise_norm_list)
    #main_evaluate_wba(net_name, loss_name, epoch, device, 'bba', loader_bba, norm_type, noise_norm_list)
    main_evaluate_wba(net_name, loss_name, epoch, device, 'test', loader_test, norm_type, noise_norm_list)
#%%
def main_train(epoch_start, epoch_end, train, arg):
#%%
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    filename=get_filename(net_name, loss_name)
    print('train model: '+filename)
    if epoch_start == epoch_end:
        print('epoch_end is epoch_start, exist main_train')
        return
    #-----------------------------------------
    device=arg['device']
    lr=arg['lr']
    if 'reset_optimizer' not in arg.keys():
        arg['reset_optimizer']=False
    reset_optimizer=arg['reset_optimizer']
    if 'batch_size' not in arg.keys():
        arg['batch_size']=128
    batch_size=arg['batch_size']
    if 'return_idx' not in arg.keys():
        arg['return_idx']=(False, False, False)
    return_idx=arg['return_idx']
    norm_type=arg['norm_type']
    #---------------------------------------
    if 'DataParallel' not in arg.keys():
        arg['DataParallel']=False
    DataParallel=arg['DataParallel']
#%%
    loader_train, loader_val, loader_test = get_dataloader(batch_size=batch_size, return_idx=return_idx)
#%%
    loss_train_list=[]
    acc_train_list=[]
    acc_val_list=[]
    acc_other_list=[]
    epoch_save=epoch_start-1
#%%
    model=Net(net_name)
    if epoch_start > 0:
        print('load', filename+'_epoch'+str(epoch_save)+'.pt')
        checkpoint=torch.load(filename+'_epoch'+str(epoch_save)+'.pt', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        #------------------------
        loss_train_list=checkpoint['result']['loss_train_list']
        acc_train_list=checkpoint['result']['acc_train_list']
        acc_val_list=checkpoint['result']['acc_val_list']
        acc_other_list=checkpoint['result']['acc_other_list']
        if 'E' in arg.keys():
            arg['E']=checkpoint['result']['arg']['E']
            print('load E')
    #------------------------
    if DataParallel == True:
        print('DataParallel on cuda', arg['device_ids'])
        torch.cuda.set_device(arg['device_ids'][0])
        model=nn.DataParallel(model, device_ids=arg['device_ids'])
        device=torch.device('cuda')
        model.to(device)
    else:
        model.to(device)
    #------------------------
    if arg['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif arg['optimizer']=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif arg['optimizer']=='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)
    elif arg['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3, nesterov=True)
    else:
        raise NotImplementedError('unknown optimizer')
    if epoch_start > 0 and reset_optimizer == False:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#%%
    if arg["scheduler"] == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, arg['epoch_end'])
    best_model = ""
    best_acc_noisy = 0        
    for epoch in range(epoch_save+1, epoch_end):
        #-------- training --------------------------------
        start = time.time()
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, arg)
        if arg["scheduler"] == 1:
            scheduler.step()
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        end = time.time()
        print('time cost:', end - start)
        #-------- validation and test --------------------------------
        acc_noisy = 0
        if epoch + 1 % 20 == 0:
            result_val = test_adv(model, device, loader_test, num_classes=10,
                                   noise_norm=8/255, norm_type=norm_type,
                                   max_iter=10, step= noise_norm/4, method='pgd')
            acc_val_list.append(result_val) 
            acc_noisy = result_val['acc_noisy']
        
        result_test = test(model, device, loader_test, num_classes=10)
        acc_other_list.append(result_test['acc'])
        #--------save model-------------------------
        result={}
        result['arg']=arg
        result['loss_train_list'] =loss_train_list
        result['acc_train_list'] =acc_train_list
        result['acc_val_list'] =acc_val_list
        result['acc_other_list'] =acc_other_list
        if not os.path.exists("result"):
            os.mkdir("result")
        if (epoch+1)%10 == 0:
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
        if acc_noisy > best_acc_noisy:
            best_acc_noisy = acc_noisy
            save_checkpoint(filename+'_best'+'.pt', model, optimizer, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        #plt.close('all')
        display.clear_output(wait=False)
        fig, ax = plot_result(loss_train_list, acc_train_list,
                              acc_val_list, acc_other_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')
        plt.close(fig)
#%%
def main_evaluate_wba(net_name, loss_name, epoch, device, data_name, loader, norm_type, noise_norm_list):
#%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model_state_dict=checkpoint['model_state_dict']
    state_dict={}
    for key, value in model_state_dict.items():
        state_dict[key.replace('module.', '')]=value
    x=loader.dataset[0][0]
    model=Net(net_name)
    model.load_state_dict(state_dict)
    model.to(x.dtype).to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    print(noise_norm_list)
    #%%
    result_100pgd = []
    result_ifgsm = []
    result_auto = []
    #%%IFGSM
    """
    print ("IFGSM is testing...")
    num_repeats=1
      
    for noise_norm in noise_norm_list:
        start = time.time()
        result_ifgsm.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                       max_iter=10, step=noise_norm/4, method='ifgsm', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
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
    
    #%% 100pgd
    """
    print ("pgd100 is testing...")
    num_repeats=2
    
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_100pgd.append(test_adv(model, device, loader, num_classes=10,
                                      noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/10, method='pgd_ce_m',
                                      num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    

    """
    """
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    result_100pgd[0]['auc']=auc
    fig, ax = plt.subplots()
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    title='wba_100pgd_L'+str(norm_type)+' r'+str(num_repeats)+' auc='+str(auc)+' '+data_name
    ax.set_title(title)
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_'+title+'.png')
    plt.close(fig)
    #%%
    """
    filename=filename+'_result_wba_L'+str(norm_type)+'_'+data_name+'.pt'
    torch.save({'result_auto':result_auto,
                'result_100pgd':result_100pgd,
                'result_ifgsm': result_ifgsm},
               filename)
    print('saved:', filename)
