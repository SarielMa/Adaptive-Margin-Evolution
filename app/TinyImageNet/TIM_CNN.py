#%%
import sys
sys.path.append('../../core')
#%%
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import time
from Evaluate import test, test_rand, cal_AUC_robustness
from Evaluate_advertorch import test_adv, test_adv_auto
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset as torch_dataset
import os
import math

#%%
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=200):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Modify for Tiny ImageNet
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model

#%% data loader
class MyDataset(torch_dataset):
    def __init__(self, data = "train", return_idx=False):
        if data == "train":
            self.data = datasets.ImageFolder(os.path.join("../../data/TIM/TIM", 'train'),
                                             transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor()]))
        elif data == "val":
            self.data = datasets.ImageFolder(
                os.path.join("../../data/TIM/TIM", 'val/images'),
                transforms.Compose([
                    transforms.ToTensor(),
                ]))           
        self.return_idx=return_idx
        
    def __len__(self):
        return len(self.data)  
      
    def __getitem__(self, idx):
        (x, y)= self.data[idx]
        if self.return_idx == False:
            return x, y
        else:
            return x, y, idx

# Data loader for Tiny ImageNet
def get_dataloader(batch_size=100, workers=4, pin_memory=True, return_idx=(False, False, False), data_aug=True):


    train_loader = torch.utils.data.DataLoader(
        MyDataset("train", return_idx[0]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        MyDataset("val", return_idx[2]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

#%%
def Net(net_name):
    if net_name == 'resnet18':
        model = resnet18()
    else:
        raise ValueError('invalid net_name ', net_name)
    return model
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def plot_result(loss_train_list, acc_train_list,
                acc_val_list, adv_acc_val_list, acc_test_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='train')
    ax[0].set_xlabel('epoch')
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='train')
    ax[1].plot(acc_val_list, '-m', label='val')
    ax[1].plot(acc_test_list, '-r', label='test')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylim(0.5, 1)
    #ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(adv_acc_val_list, '-m', label='adv val')
    ax[2].set_xlabel('epoch')
    ax[2].set_ylim(0, 0.8)
    #ax[2].legend()
    ax[2].grid(True)
    return fig, ax
#%%
def get_filename(net_name, loss_name, epoch=None, pre_fix='result/TIM_'):
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
def get_noise_norm_list(norm_type):
    if norm_type == np.inf:
        noise_norm_list=[2/255, 4/255, 6/255, 8/255]
    else:
        #noise_norm_list=[0.5, 1.0, 1.5, 2.0, 2.5]
        noise_norm_list = [0.1,0.3,0.5]
    return noise_norm_list
#%%
def main_evaluate(epoch, arg):
    net_name=arg['net_name']
    loss_name=arg['loss_name']
    device=arg['device']
    norm_type=arg['norm_type']
    noise_norm_list=get_noise_norm_list(norm_type)
    #loader_bba = get_dataloader_bba()
    loader_train, loader_test = get_dataloader()
    del loader_train
    #main_evaluate_rand(net_name, loss_name, epoch, device, loader_test, noise_norm_list)
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
    device=arg['device']
    if 'pretrained_model' not in arg.keys():
        arg['pretrained_model']='none'
    pretrained_model=arg['pretrained_model']
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
    #-----
    if 'data_aug' not in arg.keys():
        arg['data_aug']=True
    data_aug=arg['data_aug']
#%%
    if norm_type == np.inf:
        noise_norm=8/255
    elif norm_type == 2:
        noise_norm=1.0
#%%
    loader_train,loader_test = get_dataloader(batch_size=50, return_idx=return_idx, data_aug=data_aug)
#%%
    loss_train_list=[]
    acc_train_list=[]
    acc_val_list=[]
    adv_acc_val_list=[]
    acc_test_list=[]
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
        adv_acc_val_list=checkpoint['result']['adv_acc_val_list']
        if 'E' in arg.keys():
            if arg['E'] is None:
                arg['E']=checkpoint['result']['arg']['E']
                print('load E')
    elif pretrained_model != 'none':
        print('load pretrained_model', pretrained_model)
        checkpoint=torch.load(pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    #------------------------
    if DataParallel == True:
        print('DataParallel')
        torch.cuda.set_device(arg['device_ids'][0])
        model=nn.DataParallel(model, device_ids=arg['device_ids'])
        model.to(torch.device('cuda'))
    else:
        model.to(device)
    #------------------------
    if arg['optimizer']=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=arg['lr'])
    elif arg['optimizer']=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=arg['lr'], momentum=0.9, weight_decay=0.0002, nesterov=False)
    else:
        raise NotImplementedError('unknown optimizer')
    if epoch_start > 0 and reset_optimizer == False:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('load optimizer state')
        update_lr(optimizer, arg['lr'])
#%%
    best_model = ""
    best_acc_noisy = 0   
    train_time = 0
    for epoch in range(epoch_save+1, epoch_end):
        start = time.time()
        #-------- training --------------------------------
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, arg)
        train_time += (time.time() - start)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        #-------- validation --------------------------------
        if (epoch+1)%10 == 0:
            result_val = test_adv(model, device, loader_test, num_classes=200,
                                  noise_norm=noise_norm, norm_type=norm_type,
                                  max_iter=10, step=noise_norm, method='pgd')
            acc_val_list.append(result_val['acc_clean'])
            adv_acc_val_list.append(result_val['acc_noisy'])
            acc_noisy = result_val['acc_noisy']
        
        result_test=test(model, device, loader_test, num_classes=200)
        acc_test_list.append(result_test['acc'])
        #--------save model-------------------------
        result={}
        result['arg']=arg
        result['loss_train_list'] =loss_train_list
        result['acc_train_list'] =acc_train_list
        result['acc_val_list'] =acc_val_list
        result['adv_acc_val_list'] =adv_acc_val_list
        if 'E' in arg.keys():
            result['E']=arg['E']
        if not os.path.exists("result"):
            os.mkdir("result")            
        if (epoch+1)%10 == 0 :
            save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
        if (epoch+1)%10 == 0 and acc_noisy > best_acc_noisy:
            best_acc_noisy = acc_noisy
            save_checkpoint(filename+'_best'+'.pt', model, optimizer, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        fig, ax = plot_result(loss_train_list, acc_train_list,
                              acc_val_list, adv_acc_val_list, acc_test_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')
        plt.close(fig)
        end = time.time()
        print('time cost:', end - start)
        #-------check if termination is needed----------------------------------------------
        if 'termination_condition' in arg:
            if arg['termination_condition'] == arg['no_expand_times']:
                print ("termination condition is met, terminate training")
                save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
                break
    print ("================== train time: ",train_time,"=============================================")
    
#%%
def main_evaluate_wba(net_name, loss_name, epoch, device, data_name, loader, norm_type, noise_norm_list):
    #%%
    filename=get_filename(net_name, loss_name, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=torch.device('cpu'))
    model=Net(net_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    print(noise_norm_list)
    result_100pgd=[]
    result_auto=[]
    result_ifgsm=[]
    
    #%% 100pgd
    """
    num_repeats=2
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_100pgd.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='pgd_ce_cw', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('pgd100 auc is ', auc)
    
    
    #%%IFGSM
    
    num_repeats=1
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_ifgsm.append(test_adv(model, device, loader, 10, noise_norm=noise_norm, norm_type=norm_type,
                                      max_iter=100, step=noise_norm/4, method='pgd', num_repeats=num_repeats))
        end = time.time()
        print('time cost:', end - start)
    noise=[0]
    acc=[result_ifgsm[0]['acc_clean']]
    for k in range(0, len(result_ifgsm)):
        noise.append(result_ifgsm[k]['noise_norm'])
        acc.append(result_ifgsm[k]['acc_noisy'])
    auc=cal_AUC_robustness(acc, noise)
    print('fgsm auc is ', auc)
    """
    #%% auto attack
    num_repeats=1
    
    for noise_norm in noise_norm_list:
        start = time.time()
        result_auto.append(test_adv_auto(model, device, loader, 200, noise_norm=noise_norm, norm_type=norm_type,
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
    
    fig, ax = plt.subplots()
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    title='wba_100pgd_norm_type_'+str(norm_type)+'_r'+str(num_repeats)+' auc='+str(auc)+' '+data_name
    ax.set_title(title)
    ax.set_xlabel(filename)
    display.display(fig)
    
    fig.savefig(filename+'_'+title+'.png')
    plt.close(fig)
    
    #%%
    filename=filename+'_result_wba_L'+str(norm_type)+'_r'+str(num_repeats)+'_'+data_name+'.pt'
    torch.save({'result_auto':result_auto, 
                'result_100pgd_ce_cw':result_100pgd,
                'result_100pgd': result_ifgsm}, filename)
    print('saved:', filename)
