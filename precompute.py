from utils import log
import torch
import torch.nn as nn
from torchsummary import summary
import time
import csv
import numpy as np

from utils.test_utils import get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import sample_estimator, tune_mahalanobis_hyperparams
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.cal_score import *
from argparser import *

import torchvision
from torchvision import transforms
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import laplace
import torch.nn.functional as F
from tqdm import tqdm
import pickle



def get_class_mean(args, model, num_classes, train_loader):
    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        temp_x = torch.rand(2,3,32,32).cuda()
    else:
        temp_x = torch.rand(2,3,224,224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    class_mean, _ = sample_estimator(model, num_classes, feature_list, train_loader)

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)

    class_mean = class_mean[-1]

    np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)

def get_features(args, model, dataloader):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            # outputs = model.features(x)
            # feature = F.adaptive_avg_pool2d(outputs, 1)
            # feature = feature.view(feature.size(0), -1)
            # print(feature.size())
            # features.extend(feature.data.cpu().numpy())
            # features.extend(feature)
            # x = feature[feature>=0]
            # print(x.size())
            features.append(feature.cpu().numpy())

    features = np.concatenate(features, axis=0)
            

    # features = np.array(features)
    # x = np.transpose(features)
    print(features.shape)

    return features

def get_mahalanobis(args, model, num_classes, train_loader, val_loader):
    sample_mean, precision, best_regressor, best_magnitude \
        = tune_mahalanobis_hyperparams(args, model, num_classes, train_loader, val_loader)
    
    save_dir = os.path.join('cache', 'mahalanobis', args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # np.save(os.path.join(save_dir, f'{args.in_dataset}_{args.model}_results'),
    #         np.array([sample_mean.cpu(), precision.cpu(), best_regressor.coef_, best_regressor.intercept_, best_magnitude]))
        
    np.save(os.path.join(save_dir, f'{args.in_dataset}_{args.model}_results'),
            np.array([best_regressor.coef_, best_regressor.intercept_, best_magnitude]))
            
def extract_mean_std(args, model):
    
    for key, v in model.state_dict().items():
        # resnet
        if key == 'layer4.1.bn2.weight':
            # print(f'var: {v}')
            std = v
        if key == 'layer4.1.bn2.bias':
            # print(f'mean: {v}')
            mean = v
        
        if key == 'layer4.2.bn3.weight':
            # print(f'var: {v}')
            std = v
        if key == 'layer4.2.bn3.bias':
            # print(f'mean: {v}')
            mean = v

        print(f'{key}')
        if key == 'fc.weight':
            fc_w = v
            # print(v.shape)
            # print(f'fc: {v}')
        
        #wideresnet, densenet
        if key == 'bn1.weight':
            # print(f'var: {v}')
            std = v
        if key == 'bn1.bias':
            # print(f'mean: {v}')
            mean = v

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    
    torch.save(mean, f'{file_folder}/{args.model}_features_mean.pt')
    torch.save(std, f'{file_folder}/{args.model}_features_std.pt')
    # print(mean.shape)
    # print(std)
    return fc_w.cpu().numpy()

def get_LINE_info(args, model, num_classes, trainset, featdim):
    file_folder = f'cache/{args.name}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    if args.in_dataset in {'CIFAR-10', 'CIFAR-100'}:
        id_train_size = 50000

        cache_name = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_in.npy"
        if not os.path.exists(cache_name):
            shap_log = np.zeros((id_train_size, featdim))
            score_log = np.zeros((id_train_size, num_classes))
            label_log = np.zeros(id_train_size)
            
            batch_size = 1
            train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, num_workers=4)
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                
                inputs, targets = inputs.cuda(), targets.cuda()
                start_ind = batch_idx
                first_order_taylor_scores, outputs = model._compute_taylor_scores(inputs, targets)
                shap_log[start_ind, :] = first_order_taylor_scores[0].squeeze().cpu().detach().numpy()
                label_log[start_ind] = targets.data.cpu().numpy()
                score_log[start_ind] = outputs.data.cpu().numpy()
        
            np.save(cache_name, (shap_log.T, score_log.T, label_log))
            print("dataset : ", args.in_dataset)
            # print("method : ", args.method)
            print("iteration done")
        else:
            shap_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
            shap_log, score_log = shap_log.T, score_log.T
            
            shap_matrix_mean = np.zeros((featdim,num_classes))
            
            for class_num in range(num_classes):
                mask = np.array(label_log==class_num)
                masked_shap = mask[:,np.newaxis] * shap_log
                shap_matrix_mean[:,class_num] = masked_shap.sum(0) / mask.sum()
                 
            np.save(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy", shap_matrix_mean)
            print("dataset : ", args.in_dataset)
            # print("method : ", args.method)
            print("precompute done")
    else:
    ############################################################################################################
        cache_name_shap = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_in.npy"
        cache_name_score = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_score.npy"
        cache_name_label = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_label.npy"
        if not os.path.exists(cache_name_shap):
            batch_size = 1
        
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            id_train_size = len(trainset)
            
            shap_log = np.zeros((id_train_size, featdim))
            score_log = np.zeros((id_train_size, num_classes))
            label_log = np.zeros(id_train_size)
            
            model.eval()
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                
                inputs, targets = inputs.cuda(), targets.cuda()
                start_ind = batch_idx
                
                first_order_taylor_scores, outputs = model._compute_taylor_scores(inputs, targets)
                shap_log[start_ind, :] = first_order_taylor_scores[0].squeeze().cpu().detach().numpy()
                label_log[start_ind] = targets.data.cpu().numpy()
                score_log[start_ind] = outputs.data.cpu().numpy()
            
            with open(cache_name_shap, 'wb') as f:
                pickle.dump(shap_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            with open(cache_name_score, 'wb') as f:
                pickle.dump(score_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            with open(cache_name_label, 'wb') as f:
                pickle.dump(label_log.T, f, protocol=pickle.DEFAULT_PROTOCOL)
            print("dataset : ", args.dataset, "method : ", args.method, "iteration done")
        else:
            cache_name_shap = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_in.npy"
            cache_name_score = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_score.npy"
            cache_name_label = f"cache/{args.name}/{args.in_dataset}_train_{args.model}_label.npy"
            with open(cache_name_shap, 'rb') as f:
                shap_log = pickle.load(f)
            with open(cache_name_score, 'rb') as f:
                score_log = pickle.load(f)
            with open(cache_name_label, 'rb') as f:
                label_log = pickle.load(f)
            shap_log,  label_log = shap_log.T, label_log.T
            
            shap_matrix_mean = np.zeros((featdim,num_classes))

            for class_num in tqdm(range(num_classes)):
                mask = np.where(label_log==class_num)
                masked_shap = shap_log[mask[0][:]]
                num_sample = len(mask[0][:])
                shap_matrix_mean[:,class_num] = masked_shap.sum(0) / num_sample
            np.save(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy", shap_matrix_mean)
    print("done")


kwargs = {'num_workers': 2, 'pin_memory': True}
imagesize = 32
transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def main(args):
    if args.in_dataset == "CIFAR-10":
        data_path = '/data/cifar10'
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, **kwargs)
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/cifar100'
        # Data loading code
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, **kwargs)
        num_classes = 100

       
    elif args.in_dataset == "imagenet":
        root = '/data/ilsvrc2012'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), transform_test_largescale)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, **kwargs)
        num_classes = 1000

    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    
    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()

    get_class_mean(args, model, num_classes, train_dataloader)
    
    extract_mean_std(args, model)

    get_mahalanobis(args, model, num_classes, train_dataloader, val_loader)

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    features = get_features(args, model, train_dataloader)
    np.save(f"{file_folder}/{args.model}_feat_stat.npy", features.mean(0))
    
    if args.model == 'resnet18':
        featdim = 512
    # elif args.model == 'resnet50':
    #     featdim = 2048
    # elif args.model == 'mobilenet':
    #     featdim = 2048
    # elif args.model == 'densenet':
    #     featdim = 342
    get_LINE_info(args, model, num_classes, trainset, featdim)




if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    main(args)
