from utils.pruning import eval_l1_sparsity,prune, weight_transfer
import torch
import torch.nn as nn
from models.yolo import Model
import yaml
from utils.torch_utils import intersect_dicts
import argparse
import os
from utils.pruning import eval_l1_sparsity

def main(weight):
    '''
    weight:checkpoint
    '''
    ckpt = torch.load(weight, map_location="cuda:0")  # load checkpoint
    model = Model(ckpt['model'].yaml).to("cuda:0")  # create
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    sparsity = eval_l1_sparsity(model)
    print("model sparsity:",sparsity)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="prune yolov5")
    parser.add_argument('--weight',type = str ,help = 'path to yolov5 checkpoint.',default="model.pt")
    args = parser.parse_args()
    main(args.weight)