from sklearn.metrics import roc_auc_score, roc_curve
import torch, torch.nn as nn
import numpy as np

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction, multi_class='ovr')
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc


def model_parameters_count(model):
    """
    Count the model total number of parameters

    Args:
        model : (nn.Module)

    Return a dict with name of module and number of params
    """
    trainable, not_trainable = 0, 0
    for p in model.parameters():
        count = p.flatten().size()[0]
        if p.requires_grad:
            trainable += count
        else:
            not_trainable += count
    
    return dict(trainable=trainable, fixed=not_trainable)

def smooth_cross_entropy(pred, targets, p=0.0, n_classes=None):
    """
        Cross-entropy with label smoothing for slide
        
        Args:
            pred : model output (torch.Tensor)
            targets : ground-truth as index (torch.Tensor)
            p : probability for label smoothing (float)
            n_classes : number of classes (int)

        Return tensor of loss (torch.Tensor)
    """

    def smooth_labels(targets, p, n_classes):
        # produce on-hot encoded vector of targets
        # fill with:  p / (n_classes - 1)
        # fill true classes value with : 1 - p
        
        res = torch.empty(size=(targets.size(0), n_classes), device=targets.device)
        res.fill_(p /(n_classes - 1))
        res.scatter_(1, targets.data.unsqueeze(1), 1. - p)
        return res
    
    def smooth_labels_randomly(targets, p, n_classes):
        # produce on-hot encoded vector of targets
        # fill true classes value with random value in : [1 - p, 1]
        # and completes the other to sum up to 1
        
        res = torch.zeros((targets.size(0), n_classes), device=targets.device)
        rand = 1 - torch.rand(targets.data.unsqueeze(1).shape, device=targets.device)*p
        res.scatter_(1, targets.data.unsqueeze(1), rand)
        fill_ = (1 - res.sum(-1))/(n_classes - 1)
        return res.maximum(fill_.unsqueeze(dim=-1).repeat(1, n_classes))

    assert isinstance(pred, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    
    if p:
        if n_classes is None:
            n_classes = pred.size(-1)
            
        targets = smooth_labels_randomly(targets, p, n_classes)
        pred = pred.log_softmax(dim=-1)
        cce_loss = torch.sum(-targets * pred, dim=1)
        return torch.mean(cce_loss)
    else:
        return nn.functional.cross_entropy(pred, targets.to(dtype=torch.long))
