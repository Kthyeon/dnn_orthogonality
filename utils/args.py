import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train script')
    
    # seed and gpu
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--root', default='/home/taehyeon/', type=str, help='dataset path')
    
    # data setter
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'tiny-image', 'image'] , help='dataset')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes in dataset')
    parser.add_argument('--valid_size', default=0.1, type=float, help='portion for valid data. 0.0 indicates no valid data')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--pin_memory', action='store_true', help='whether pin memory on data loader')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    # model : resnet
    parser.add_argument('--model', default='resnet-50', type=str, help='name of the model to train')
    parser.add_argument('--opt', default='both', type=str, choices=['none', 'exp', 'rec', 'both'], help='position for ortho reg of pointwise conv')
    parser.add_argument('--init', default='xavier', type=str, choices=['xavier','kaiming','ort','z_ort'], help='initialization for network')
    parser.add_argument('--pre_trained', action='store_true', help = 'whether use pretrained model or not')

    # train & test phase
    parser.add_argument('--optim', default='sgd', type=str, help='optimization function', choices=['sgd', 'rmsprop', 'adam', 'adagrad', 'adadelta'])
    parser.add_argument('--lr_sch', default='step', type=str, choices=['step', 'cos', 'exp'], help='lr scheduler (cos/step)')
    parser.add_argument('--warmstart', action='store_true', help='whether warm start or not')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda:0', type=str, help='which GPU to use')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--milestones', default='100,150', type=str, help='milestone epochs')
    parser.add_argument('--gamma', default=0.1, type=float, help='scheduler ratio')
    parser.add_argument('--criterion', default='ce', type=str, choices=['ce', 'ls'], help='loss objective function')
    parser.add_argument('--eps', default=0.1, type=float, help='smoothing epsilon for label smoothing')

    # regularization
    parser.add_argument('--wd_ablation', action='store_true', help='whether to anatomize')
    parser.add_argument('--inreg', default='none', type=str, choices=['none', 'cutmix', 'mixup'], help='input regularization')
    parser.add_argument('--ortho', default='none', type=str, choices=['none','norm','srip','ort','ortho','inputnorm', 'downinnorm', 'sin_srip'], help='Orthogonal regularization')
    parser.add_argument('--lamb_list', default='0.0_1.0_0.0_0.0', type=str, help='lambda for each class of filter. [origin, point, depth, fully connected]')
    parser.add_argument('--tp', default='app', type=str, choices=['app', 'ori'], help='orthogonal regularization on depthwise convolution')

    return parser.parse_args()