import logging

import torch
from torch.optim import lr_scheduler
from torch import cuda, nn, optim
from tqdm import tqdm, trange
from torch.nn.functional import normalize
from torch.nn.utils import clip_grad_norm_

import copy
import numpy as np
import time
import pandas as pd

from utils.cutmix import rand_bbox
from regularization import *
from utils.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from models import *

# Here, we have to add logger for train loss, accuracy etc.
# Using csv file.

def load_model(model_name, args):
    # name consists of '-' whose components are depth, width, and class of network.
    name_list = model_name.split('-')
    try:
        if name_list[0] == 'effb0':
            # only here effb0
            model = EfficientNetB0(opt = args.opt, init = args.init, pre_trained=args.pre_trained, num_classes=args.num_classes)
        elif name_list[0] == 'micr':
            # micr-1-2-3: 1 is wide_factor, 2 is depth_factor, and 3 is expansion, respectively.
            wide_f = float(name_list[1]) 
            depth_f = float(name_list[2])
            expan = float(name_list[3])
            model = MicroNet(opt = args.opt, init = args.init, num_classes = args.num_classes, wide_factor = wide_f, depth_factor = depth_f, expansion = expan)
        elif 'vgg' in model_name:
            if model_name == 'vgg11':
                model = vgg11(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg13':
                model = vgg13(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg16':
                model = vgg16(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg19':
                model = vgg19(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg11-bn':
                model = vgg11_bn(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg13-bn':
                model = vgg13_bn(init = args.init, num_classes = args.num_classes)
            elif model_name == 'vgg16-bn':
                 model = vgg16_bn(init = args.init, num_classes = args.num_classes)
            if model_name == 'vgg19-bn':
                 model = vgg19_bn(init = args.init, num_classes = args.num_classes)
        elif name_list[0] == 'res':
            # res-50-1-t-bn or nobn
            print(name_list)
            depth = float(name_list[1])
            width = int(name_list[2])
            bottlen = True if name_list[3]=='t' else False
            batchn = True if name_list[4] == 'bn' else False
            model = ResNet(opt = args.opt, init = args.init, num_classes = args.num_classes, batchnorm = batchn, width = width, depth = depth, bottleneck= bottlen)
        elif name_list[0] == 'wrn':
            # wrn-28-4
            depth = int(name_list[1])
            width = int( name_list[2])
            model = WideResNet(opt = args.opt, init = args.init, num_classes = args.num_classes, widen_factor = width, depth = depth)
        elif name_list[0] == 'mobv2':
            # mobv2-1.0 : 1.0 width - # of channels
            width = float(name_list[1])
            model = mobilenetv2(opt = args.opt, init = args.init, num_classes = args.num_classes, width_mult=width)
        elif name_list[0] == 'mobv3':
            # model : mobv3-large-1.0
            width = float(name_list[2])
            if name_list[1]  == 'large':
                model = mobilenetv3_large(opt = args.opt, init = args.init, num_classes = args.num_classes, width_mult = width)    
            else:
                model = mobilenetv3_small(opt = args.opt, init = args.init, num_classes = args.num_classes, width_mult = width)
        else:
            raise Exception('No option for this model')
        return model

    except Exception as e:
        print(e)

    
def make_log(model_name, args):
    log_filename = './results/' + args.dataset + '/' +  model_name + '/' + args.optim + '/' + args.lr_sch + '/' + args.init + '_' + args.opt + '_' + args.ortho + '_' + args.lamb_list + '_' + args.tp  + '_seed'  + str(args.seed) +  '_log.csv'
    log_columns = ['train_loss', 'train_accuracy']

    if args.valid_size !=0:
        log_columns += ['valid_loss', 'valid_accuracy']
    log_columns += ['test_loss', 'test_accuracy']
    log_pd = pd.DataFrame(np.zeros([args.num_epochs, len(log_columns)]), columns = log_columns)

    return log_filename, log_pd


def train(model, dataloader, args):
    # dataloader: dictionary type, keys: ['train', 'valid', test']
    # Basic hyperparameters setting
    # Device, momentum, learning rate, total epoch, milestones, gamma, weight decay, nesterov.
    device = args.device
    momentum = args.momentum
    learning_rate = args.lr
    num_epochs = args.num_epochs
    milestones = args.milestones
    gamma = args.gamma
    weight_decay = args.weight_decay
    nesterov = args.nesterov
    
    milestones = milestones.split(',')
    milestones = [int(mile) for mile in milestones]
    
    first_wts = None

    # Set criterion
    # Either crossentropy or labelsmoothing.  
    # handle this with 'args.criterion'. It is in ['cr', 'ls'].
    if args.criterion == 'ls':
        criterion = LabelSmoothing(args.device, args.num_classes, smoothing=args.eps, dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Parameters of network.
    batch_params = [module for name, module in model.named_parameters() if module.ndimension() == 1 and 'bn' in name]
    other_params = [module for module in model.parameters() if module.ndimension() > 1]
    
    # Optimizer for training.
    if args.optim == 'sgd': 
        optimizer = torch.optim.SGD([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': 0}], lr=learning_rate, momentum=momentum, nesterov=nesterov)
    elif args.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': 0}], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=momentum, centered=False)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': 0}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    elif args.optim == 'adadelta':
        optimizer = torch.optim.Adadelta([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': 0}], lr = 1.0, rho = 0.9, eps=1e-6)
    elif args.optim == 'adagrad':
        optimizer = torch.optim.Adagrad([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': 0}], lr = 0.01, lr_decay = 0, initial_accumulator_value=0, eps=1e-10)
    
    
    # Learning rate scheduler. It is in ['step', 'cos'].
    if args.lr_sch == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = num_epochs, eta_min=0.0005, last_epoch=-1)
    elif args.lr_sch == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma = gamma, last_epoch=-1)
    else:
        scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    
    best_acc = 0.
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # make log file
    log_filename, log_pd = make_log(model_name = args.model, args=args)
    
    lambda_list = args.lamb_list.split('_')
    lambda_list = [float(lamb) for lamb in lambda_list]

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        
        epoch_log = []
        top1 = AverageMeter()
        losses = AverageMeter()
        
        for i, (images, labels) in enumerate(tqdm(dataloader['train'])):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            if args.inreg != 'none':
                if args.inreg == 'cutmix':
                    lam, images, labels_a, labels_b = cutmix(images, labels, device)
                elif args.inreg == 'mixup':
                    lam, images, labels_a, labels_b = mixup(images, labels, device)
                optimizer.zero_grad()

                outputs = model(images)

                loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            if args.ortho == 'norm':
                loss += norm_reg(mdl = model, device = device, lamb_list = lambda_list, opt = args.opt)
            elif args.ortho == 'srip':
                loss += srip_reg(mdl = model, device = device, lamb_list = lambda_list, opt = args.opt, tp = args.tp)
            elif args.ortho == 'ort':
                loss += or_reg(mdl = model, device = device, lamb_list = lambda_list, opt = args.opt)
            elif args.ortho == 'noise':
                loss += noise_reg(mdl = model, device = device, lamb_list = lambda_list, opt = args.opt)
            elif args.ortho == 'inputnorm':
                loss += lambda_list[0] * model.make_norm_dif(images)
            elif args.ortho == 'downinnorm':
                loss += lambda_list[0] * model.make_norm_dif(images, down = True)
                
            # assign weight decay to parameters which is not penalized via ORN.
            if args.ortho == 'none':
                lambda_list = [0.0, 0.0, 0.0, 0.0]
            loss += weight_decay * wd_reg(mdl=model, device =device, lamb_list=[1.0 if lamb==0.0 else 0.0 for lamb in lambda_list])
            

            prec1, _ = accuracy(outputs.data, labels.data, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))

            loss.backward()
            optimizer.step()
                        
        epoch_log += [round(losses.avg, 4), round(top1.avg, 4)] 
        if args.valid_size != 0:
            valid_accuracy, valid_loss = eval_(model, dataloader['valid'], args)
            epoch_log += [round(valid_loss,4), round(valid_accuracy, 4)]
        test_accuracy, test_loss = eval_(model, dataloader['test'], args)
        epoch_log += [round(test_loss, 4), round(test_accuracy, 4)]
        log_pd.loc[epoch] = epoch_log
        log_pd.to_csv(log_filename)
        
        if epoch==0:
            first_wts = copy.deepcopy(model.state_dict())

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            for k, v in best_model_wts.items():
                best_model_wts[k] = v.cpu()

    return  first_wts, best_model_wts

def eval_(model, test_loader, args):
    device = args.device
    if args.criterion == 'ls':
        criterion = LabelSmoothing(args.device, args.num_classes, smoothing=args.eps, dim=1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    for i, data in enumerate(tqdm(test_loader)):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        pred_label = model(image)

        loss = criterion(pred_label, label)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_label.data, label.data, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    acc = top1.avg
    loss = losses.avg

    return acc, loss


def train_image(model, dataloader, test_loader, args):
    device = model.device
    momentum = model.momentum
    learning_rate = model.lr
    num_epochs = model.num_epochs
    milestones = model.milestones
    gamma = model.gamma
    weight_decay = model.weight_decay
    nesterov = model.nesterov
    if args.label_regularize == 'labelsmooth':
        criterion = LabelSmoothingLoss(model.device, model.num_classes, 0.1, 1)
    else:
        criterion = model.criterion
    batch_number = len(dataloader.dataset) // dataloader.batch_size    
    

    batch_params = [module for module in model.parameters() if module.ndimension() == 1]
    other_params = [module for module in model.parameters() if module.ndimension() > 1]
    optimizer = torch.optim.SGD([{'params' : batch_params, 'weight_decay': 0},
            {'params': other_params, 'weight_decay': weight_decay}], lr=learning_rate, momentum=momentum, nesterov=nesterov)
        
    if args.lr_type == 'step':
        scheduler = lr_scheduler.MultiStepLR(gamma=gamma, milestones=milestones, optimizer=optimizer)
    elif args.lr_type == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max = num_epochs, eta_min=args.min_lr, last_epoch=-1)
    losses = []
    test_losses = []
    accuracies = []
    test_accuracies = []
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    #scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=100, total_epoch=5, after_scheduler=scheduler)
    

    for epoch in range(num_epochs):
        model.train()
        #scheduler_warmup.step()
        scheduler.step()
        
        if args.label_regularize == 'labelsimilar':
            similarity = fc_similarity(model, device)
            criterion = LabelSimilarLoss(model.device, args.num_classes, similarity, 0.1, 1)
            
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            
            optimizer.zero_grad()
            
            if args.inreg != 'None':
                if args.inreg == 'cutmix':
                    lam, images, labels_a, labels_b = cutmix_32bit(images, labels, device)
                elif args.inreg == 'mixup':
                    lam, images, labels_a, labels_b = mixup_32bit(images, labels, device)


                outputs = model(images)

                loss = lam * criterion(outputs, labels_a) + (1-lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            if args.ortho == 'True':
                loss += args.ortho_lr * l2_reg_ortho_32bit(model, device)


            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            
            
            if (i + 1) % (batch_number // 10) == 0:
                tqdm.write('Epoch[{}/{}] , Step[{}/{}], Loss: {:.4f}, lr = {}'.format(epoch + 1,
                                                                                                          num_epochs,
                                                                                                          i + 1, len(
                        dataloader), loss.item(), optimizer.param_groups[0]['lr']))
            
            
        print('|| Val : Epoch {} / {} ||'.format(epoch, num_epochs))
        test_accuracy, test_loss = eval_(model, test_loader)
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, './Checkpoint/' + 'imagenet_test' + '.t7')
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

    return losses, accuracies, test_losses, test_accuracies, best_model_wts

