import os
import time
import shutil
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from counting import count

from utils import *
from models import *

# parser argument 바뀐것 들 모두다 check (다 추가하기.)
# 파일의 저장형태 확인하기.
# models 속의 다른 model들도 형태 바꾸기.





def save_checkpoints(state, name):
    torch.save(state, name)
    print(name)
    

if __name__ == '__main__':
    args = parse_args()
    torch.autograd.set_detect_anomaly(True)
    
    # whether fix the seed or not.
    if args.seed is not None:
        # make experiment reproducible
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # data setter
    dataloader, dataset_size = data_setter(root=args.root, args=args)

    # make dir (path) for results, checkpoint.
    for tmp_path in ['./results', './checkpoint']:
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        next_path = os.path.join(tmp_path, args.dataset)
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, args.model)
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, args.optim)
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, args.lr_sch)
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
    
    # gpu
    try:
        if torch.cuda.is_available():
            device = torch.device(args.device)
            print(device)
        else:
            raise Exception('gpu is not available')
    except Exception as e:
        print(e)

    # load model
    net = load_model(args.model, args=args).to(device)
    # init checkpoint
    init_model_wts = copy.deepcopy(net.state_dict())
    
    # train and eval
    first_wts, best_model_wts = train(net, dataloader, args)
    
    state = {}
    state['init_wts'] = init_model_wts
    state['first_wts'] = first_wts
    state['best_wts'] = best_model_wts
    
    save_checkpoints(state, './checkpoint/' + args.dataset + '/' + args.model + '/' + args.optim + '/' + args.lr_sch + '/' + args.init + '_' + args.opt + '_' + args.ortho + '_'  + args.lamb_list + '_seed' + str(args.seed) + '_' + str(args.wd_ablation) + '.pt')
