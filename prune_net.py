from utils.pruning import eval_l1_sparsity,prune, weight_transfer
import torch
import torch.nn as nn
from models.yolo import Model
import yaml
from utils.torch_utils import intersect_dicts
import argparse
import os

def prune_net(weight,save_dir="pruned_net",pruning_ratio=0.5,device="cuda:0"):
    '''
    weight:checkpoint
    '''
    ckpt = torch.load(weight, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml).to(device)  # create
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    sparsity = eval_l1_sparsity(model)
    print("model sparsity:",sparsity)
    mask,new_cfg = prune(model,pruning_ratio)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg_name="{}_{}_pruned_net.yaml".format("prune",pruning_ratio)

    with open (os.path.join(save_dir,cfg_name),"w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)


    model_pruned = Model(new_cfg).to(device)
    new_model = weight_transfer(model,model_pruned,mask)

    # test new_model forward
    inputs = torch.rand(1,3,640,640).cuda()
    new_model.eval()
    new_model.cuda()
    outputs_pruned = new_model.forward(inputs)
    print(outputs_pruned[0].shape)

    #save new model
    ckpt["model"]=new_model
    ckpt["best_fitness"]=0.0
    model_name = "{}_{}_pruned_net.pt".format("prune",pruning_ratio)
    torch.save(ckpt,os.path.join(save_dir,model_name))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="prune yolov5")
    parser.add_argument('--weight',type = str ,help = 'path to yolov5 checkpoint.')
    parser.add_argument('--save_dir',type = str ,help = 'path to save output files.',default="pruned_net")
    parser.add_argument('--pruning_ratio',type = float ,help = 'pruning ratio',default=0.3)
    args = parser.parse_args()
    prune_net(args.weight,args.save_dir,args.pruning_ratio)