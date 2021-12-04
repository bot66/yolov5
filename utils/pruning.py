from copy import deepcopy
import torch
from torch.distributed.distributed_c10d import gather
import torch.nn as nn
from torch.nn.modules import module
import yaml
from models.common import *
from models.experimental import *
import numpy as np


class SPPPruned(nn.Module):
    def __init__(self, c1, c2,c_, k=(5, 9, 13)):
        super().__init__()
        self.cv1 = Conv(c1,c_,1,1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self,x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class C3Pruned(nn.Module):
    def __init__(self, c1, c2, c_1,c_2,c_bk,\
                    n=1, shortcut=True, g=1, e=0.5,\
                    ): #c_bk[i]=(bk.cv1.c2,bk.cv2.c2)
        super().__init__()
        self.cv1 = Conv(c1, c_1, 1, 1)
        self.cv2 = Conv(c1, c_2, 1, 1)
        _ = []
        bk_c1 = c_1
        for i in range(n):
            _.append(BottleneckPruned(bk_c1,c_bk[i][1],c_bk[i][0]))
            bk_c1 = c_bk[i][1]
        self.cv3 = Conv(bk_c1+c_2, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*_)
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class BottleneckPruned(nn.Module):
    def __init__(self, c1, c2, c_,shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def eval_l1_sparsity(model, reduction="sum"):
    bn_gamma = []
    for k, v in model.named_modules():
        if isinstance(v, nn.BatchNorm2d):
            bn_gamma.append(v.weight)
    gammas = torch.cat(bn_gamma)
    _ = torch.zeros_like(gammas)
    l1_gamma = nn.SmoothL1Loss(reduction=reduction)
    gamma_sparsity = l1_gamma(gammas,_).detach()

    return gamma_sparsity
    

# additional subgradient descent on the sparsity-induced penalty term
# https://github.com/foolwood/pytorch-slimming/blob/master/main.py
def updateBN(model,s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

@torch.no_grad()
def prune(model,ratio=0.7):
    cfg = model.yaml
    new_cfg = deepcopy(cfg)
    new_cfg["is_pruned"] = True
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * ratio)
    thre = y[thre_index]

    pruned = 0
    cfg_mask = []
    c2 =1
    c3_mask_wrap = []
    spp_mask_wrap =[]
    for i,m in enumerate(model.model):
        module_type = type(m).__name__
        print(module_type)
        if module_type not in ["Concat","Upsample","Detect"]:
            c2s = []
            for n,sub_m in m.named_modules():
                if isinstance(sub_m, nn.BatchNorm2d):
                    weight_copy = sub_m.weight.data.clone()
                    mask = weight_copy.abs().gt(thre.cuda()).float().cuda()
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    sub_m.weight.data.mul_(mask)
                    sub_m.bias.data.mul_(mask)
                    c2 = torch.sum(mask)
                    if module_type=="C3":
                        c3_mask_wrap.append(mask.clone()) # wrap c3 masks,[cv1,cv2,cv3,bk1.cv1,bk1.cv2...,bkn.cv2]
                        c2s.append(int(c2.item()))      
                        continue
                    if module_type=="SPP":
                        spp_mask_wrap.append(mask.clone()) # wrap c3 masks,[cv1,cv2,cv3,bk1.cv1,bk1.cv2...,bkn.cv2]
                        c2s.append(int(c2.item()))      
                        continue
                    cfg_mask.append(mask.clone())  
                    c2s.append(int(c2.item()))

            if c3_mask_wrap:
                cfg_mask.append(c3_mask_wrap)
                c3_mask_wrap = []    
            if spp_mask_wrap:
                cfg_mask.append(spp_mask_wrap)
                spp_mask_wrap = []

            if i < len(new_cfg["backbone"]):
                k, idx= "backbone",i
            else:
                k, idx= "head",i-len(new_cfg["backbone"])

            if len(c2s)==1:
                new_cfg[k][idx][-1][0]= c2s[0]
            else:
                if module_type=="C3":
                    c_bk = np.array(c2s[3:]).reshape(-1,2).tolist()
                    args = new_cfg[k][idx][-1]

                    args[0] = c2s[2]

                    new_args = args[:1] + c2s[:2]
                    new_args.append(c_bk)
                    new_args = new_args + args[1:]
                    new_cfg[k][idx][-1]=new_args
                    new_cfg[k][idx][2]="C3Pruned"
        
                elif module_type=="SPP":
                    new_cfg[k][idx][-1][0]=c2s[1]

                    new_cfg[k][idx][-1].insert(1,c2s[0])
                    new_cfg[k][idx][2]="SPPPruned"
        else:
            cfg_mask.append([])

    return cfg_mask,new_cfg



def weight_transfer(model,new_model,mask):
    start_mask = torch.ones(12) #Focus layer c_in=3*4
    mask_idx = 0
    end_mask = mask[mask_idx]
    for m0,m1 in zip(model.model,new_model.model):
        module_type = type(m0).__name__
        print(module_type)
        c3_idx =0
        if "C3" in module_type: #cv1, cv2, bk1.cv1, bk1.cv2...bkn.cv1, bkn.cv2
            for i,(sub_m0,sub_m1) in enumerate(zip(m0.modules(),m1.modules())):
                if isinstance(sub_m0, nn.BatchNorm2d):

                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask[c3_idx].cpu().numpy())),axis=1)
                    sub_m1.weight.data = sub_m0.weight.data[idx1].clone()
                    sub_m1.bias.data = sub_m0.bias.data[idx1].clone()
                    sub_m1.running_mean = sub_m0.running_mean[idx1].clone()
                    sub_m1.running_var = sub_m0.running_var[idx1].clone()

                    if c3_idx<1: #cv1->cv2
                        pass
                    elif c3_idx==1: # concat(bkn.cv2,cv2)->cv3
                        start_mask = torch.cat([end_mask[-1].clone(),end_mask[1].clone()])
                    elif c3_idx==2:
                        start_mask = end_mask[0] #cv1->bk1.cv1
                    elif c3_idx>2 and c3_idx<len(end_mask)-1:
                        start_mask = end_mask[c3_idx]
                    else:
                        mask_idx += 1
                        start_mask = end_mask[2]
                        end_mask = mask[mask_idx]

                    c3_idx+=1

                elif isinstance(sub_m0, nn.Conv2d):        
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())),axis=1)
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask[c3_idx].cpu().numpy())),axis=1)
                    print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                    w = sub_m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    sub_m1.weight.data = w.clone()               
        elif "SPP" in module_type: #cv1, cv2, bk1.cv1, bk1.cv2...bkn.cv1, bkn.cv2
            layer_level = 0
            for i,(sub_m0,sub_m1) in enumerate(zip(m0.modules(),m1.modules())):
                if isinstance(sub_m0, nn.BatchNorm2d):

                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask[layer_level].cpu().numpy())),axis=1)
                    sub_m1.weight.data = sub_m0.weight.data[idx1].clone()
                    sub_m1.bias.data = sub_m0.bias.data[idx1].clone()
                    sub_m1.running_mean = sub_m0.running_mean[idx1].clone()
                    sub_m1.running_var = sub_m0.running_var[idx1].clone()

                    
                    if layer_level==0:
                        start_mask = torch.cat([end_mask[layer_level].clone() for i in range(4)]) # 1+3 concat
                        layer_level+=1
                    else:
                        start_mask = end_mask[layer_level].clone()
                        mask_idx += 1
                        end_mask = mask[mask_idx]

                elif isinstance(sub_m0, nn.Conv2d):        
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())),axis=1)
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask[layer_level].cpu().numpy())),axis=1)
                    
                    print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                    w = sub_m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    sub_m1.weight.data = w.clone()
        elif module_type=="Concat":
            f=m0.f
            right_mask = mask[f[1]]
            if isinstance(right_mask,list):
                right_mask=right_mask[2]
            start_mask=torch.cat([start_mask,right_mask])
            mask_idx += 1
            end_mask = mask[mask_idx]
        elif module_type=="Upsample":
            mask_idx += 1
            end_mask = mask[mask_idx]
        elif module_type=="Detect":
            f=m0.f
            start_masks=[]
            for i in f:
                start_mask=mask[i]
                if isinstance(start_mask,list):
                    start_mask=start_mask[2]
                start_masks.append(start_mask)                
            j=0
            for sub_m0,sub_m1 in zip(m0.modules(),m1.modules()):
                if isinstance(sub_m0, nn.Conv2d):             
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_masks[j].cpu().numpy())),axis=1)
                    print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], 18))
                    w = sub_m0.weight.data[:, idx0, :, :].clone()
                    sub_m1.weight.data = w.clone()
                    j+=1  
        else:
            for sub_m0,sub_m1 in zip(m0.modules(),m1.modules()):
                if isinstance(sub_m0, nn.BatchNorm2d):            
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())),axis=1)
                    sub_m1.weight.data = sub_m0.weight.data[idx1].clone()
                    sub_m1.bias.data = sub_m0.bias.data[idx1].clone()
                    sub_m1.running_mean = sub_m0.running_mean[idx1].clone()
                    sub_m1.running_var = sub_m0.running_var[idx1].clone()
                    mask_idx += 1
                    start_mask = end_mask.clone()
                    end_mask = mask[mask_idx]
                elif isinstance(sub_m0, nn.Conv2d):              
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())),axis=1)
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())),axis=1)
                    print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                    w = sub_m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    sub_m1.weight.data = w.clone()

    return new_model       