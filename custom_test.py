import argparse
import json
import os
import pandas
import glob
import math
import pickle
import random
import numpy as np
import tqdm
from PIL import Image
from sklearn import metrics

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import torchvision.transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')

from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
from Model.network import    Classifier_1fc, DimReduction
from utils import eval_metric

def metrics_fn(Y_true, Y_pred, class_idx):
    """
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    class_idx : dict of class index
    """

    # cross-entropy error
    error = metrics.log_loss(Y_true, Y_pred)
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

    # computes binary metrics for each class (one versus all)
    for k, i in class_idx.items():
        
        # statistics from sklearn confusion matrix
        tn, fp, fn, tp = multiclass_cm[i].ravel()

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
        fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        metrics_.update({
            "{}_precision".format(k): precision,
            "{}_recall".format(k): recall,
            "{}_fscore".format(k): fscore,
            "{}_fnr".format(k): fnr,
            })

    return metrics_


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = pandas.read_csv(params.labels).set_index('patient_id')['label'].to_dict()

    # model
    in_chn = 1024
    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(device=device)
    attention = Attention(params.mDim).to(device=device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(device=device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(device=device)

    state_dict = torch.load(os.path.join(params.model_dir, 'best_model.pth'), map_location='cpu')
    classifier.load_state_dict(state_dict['classifier'])
    attention.load_state_dict(state_dict['attention'])
    dimReduction.load_state_dict(state_dict['dim_reduction'])
    attCls.load_state_dict(state_dict['att_classifier'])
    
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()

    instance_per_group = params.total_instance // params.numGroup

    result_per_patient = {}
    for p in tqdm.tqdm(os.listdir(params.mDATA_dir_test0), ncols=100):
        patches = glob.glob(os.path.join(params.mDATA_dir_test0, p, "*", "*.pt"))
        tfeat = torch.stack(list(map(torch.load, patches))).to(device=device)
            
        with torch.no_grad():
            midFeat = dimReduction(tfeat)

            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

            allSlide_pred_softmax = []

            for jj in range(params.num_MeanInference):

                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), params.numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                slide_d_feat = []
                slide_sub_preds = []

                for tindex in index_chunk_list:
                    idx_tensor = torch.LongTensor(tindex).to(device=device)
                    tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                    tAA = AA.index_select(dim=0, index=idx_tensor)
                    tAA = torch.softmax(tAA, dim=0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                    # tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                    # slide_sub_preds.append(tPredict)

                    # patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                    # patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    # patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                    # _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                    # if params.distill_type == 'MaxMinS':
                    #     topk_idx_max = sort_idx[:instance_per_group].long()
                    #     topk_idx_min = sort_idx[-instance_per_group:].long()
                    #     topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    #     d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    #     slide_d_feat.append(d_inst_feat)
                    # elif params.distill_type == 'MaxS':
                    #     topk_idx_max = sort_idx[:instance_per_group].long()
                    #     topk_idx = topk_idx_max
                    #     d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    #     slide_d_feat.append(d_inst_feat)
                    # elif params.distill_type == 'AFS':
                    #     slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat.append(tattFeat_tensor)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                gSlidePred = attCls(slide_d_feat)
                allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

        allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
        y_pred = torch.mean(allSlide_pred_softmax, dim=0).to('cpu').numpy()
        
        # duplicate patient label to all samples
        true_idx = slide_class_idx[labels[int(p)]]
        y_true = np.zeros(len(slide_class_idx), dtype='int32')
        y_true[true_idx] = 1
        
        # add to dict
        result_per_patient[p] = (y_true, y_pred)
    
    # calculates metrics across patches
    y_true, y_pred = zip(*result_per_patient.values())
    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    patches_metrics = metrics_fn(y_true, y_pred, slide_class_idx)
    
    df = pandas.DataFrame([patches_metrics])
    
    # round floats to 2 decimals
    df = df.round(decimals=2)

    # save results in a CSV file
    df.to_csv(params.output, sep=';')


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--labels', required=True, type=str)
parser.add_argument('--model_dir', required=True, type=str, help='path to model weights directory')
parser.add_argument('--output', required=True, type=str, help='path to the CSV file to save results')
parser.add_argument('--num_cls', required=True, type=int)
parser.add_argument('--mDATA_dir_test0', required=True, type=str)         ## Test Set
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
params = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)

slide_class_idx = {'ABC': 1, 'GCB': 0}

if __name__ == "__main__":
    main(params)