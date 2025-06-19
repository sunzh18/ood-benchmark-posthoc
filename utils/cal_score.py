from utils import log
import torch
import time

import numpy as np

import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
import torch.nn.functional as F

def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_HIMPLoS(data_loader, model, temper, mask, p, threshold, class_mean):

    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            output = model(x)      

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_prototype = torch.zeros(feature.shape).cuda()

            cp = feature * mask[pred_y,:].cuda()
            class_prototype = class_mean[pred_y,:].cuda() 

            cos_sim = F.cosine_similarity(class_prototype, feature, dim=1)

            cp = cp.clip(max=threshold)
            
            logits = model.forward_head(cp)
            logits = logits * cos_sim[:, None]

            conf = temper * (torch.logsumexp((logits) / temper, dim=1))

            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        # tempInputs = torch.add(x.data, gradient, -epsilon)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        # if b % 100 == 0:
        #     logger.info('{} batches processed'.format(b))
        # debug
        # if b > 500:
        #    break

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_react(data_loader, model, temper, threshold):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model.forward_threshold(x, threshold)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_LINE(data_loader, model, temper, threshold):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model.forward_LINE(x, threshold)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_dice(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 10 == 0:
        #     logger.info('{} batches processed'.format(b))
        x = x.cuda()
        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)

def iterate_data_featurenorm(data_loader, model):

    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            local_feature, _ = model.encoder.local_features(x)# [bz, 512] # 特征提取器得到的feature
            local_feature = local_feature.transpose(1, 2)
            # print(local_feature.shape)
            feature_l2norm = torch.norm(local_feature, p=2, dim=2)
            # print(feature_l2norm.shape)
            score = feature_l2norm.mean(1)
            # print(score.shape)
            conf = score

            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def bats_iterate_data_energy(data_loader, model, temper, lam=None, feature_std=None, feature_mean=None, bats=False):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            features = model.forward_features(x)
            # print(features.size())
            # f = model.features(x)
            # print(f.size())
            if bats:
                features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
                features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            logits = model.forward_head(features)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)