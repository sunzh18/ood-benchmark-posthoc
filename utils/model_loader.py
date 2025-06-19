import os

from models.cifar_resnet import resnet18_cifar, resnet34_cifar
from models.resnet_react import resnet18, resnet50
from models.mobilenet import mobilenet_v2
from models.densenet import DenseNet3
import torch


on_gpu = torch.cuda.is_available() 


def get_model(args, num_classes, load_ckpt=True, info=None, LU=False):
    if args.in_dataset == 'imagenet':
        checkpoint = None
        if args.model == 'resnet18':
            model = resnet18(num_classes=num_classes, pretrained=True, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
            
        elif args.model == 'resnet50':
            model = resnet50(num_classes=num_classes, pretrained=True, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)

        elif args.model == 'mobilenet':
            model = mobilenet_v2(num_classes=num_classes, pretrained=True, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)

        if checkpoint:
            model.load_state_dict(checkpoint)
    else:
        if args.model == 'resnet18':
            model = resnet18_cifar(num_classes=num_classes, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'resnet34':
            model = resnet34_cifar(num_classes=num_classes, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'densenet':
            model = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model)

    if on_gpu:                                                   #部署到GPU上
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    model = model.to(device) 

    if (args.in_dataset != 'imagenet') and load_ckpt:
        checkpoint = torch.load(f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth')
        model.load_state_dict(checkpoint['state_dict'])


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model