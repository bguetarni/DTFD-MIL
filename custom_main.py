import argparse, json, os, glob, random
import pandas
import numpy as np
import tqdm
import sklearn

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
from Model.network import    Classifier_1fc, DimReduction
from utils import eval_metric

def metrics_fn(args, Y_true, Y_pred):
    """
    args : script arguments
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    """

    # cross-entropy error
    error = sklearn.metrics.log_loss(Y_true, Y_pred)

    # ROC AUC (per class)
    auc = dict()
    for i in range(Y_true.shape[1]):
        # select class one-hot values
        ytrue = Y_true[:,i]

        # transform probabilities from [0.5,1] to [0,1]
        # probabilities in [0,0.5] are clipped to 0
        ypred = np.clip(Y_pred[:,i], 0.5, 1) * 2 - 1
        auc_score = sklearn.metrics.roc_auc_score(ytrue, ypred)
        auc.update({i: auc_score})
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = sklearn.metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = sklearn.metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = sklearn.metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = sklearn.metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = sklearn.metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = sklearn.metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

    # computes binary metrics for each class (one versus all)
    for k, i in TRAIN_PARAMS['class_to_label'][args.dataset].items():
        
        # statistics from sklearn confusion matrix
        tn, fp, fn, tp = multiclass_cm[i].ravel()

        # metrics
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fnr = fn / (fn + tp)
        fscore = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        
        metrics_.update({
            "{}_auc".format(k): auc[i],
            "{}_precision".format(k): precision,
            "{}_recall".format(k): recall,
            "{}_fscore".format(k): fscore,
            "{}_fnr".format(k): fnr,
            })

    return metrics_

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)

def load_data(args, validation_factor=0.2):

    if args.dataset in ["chulille", "dlbclmorph"]:
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    data = []
    if args.dataset == "chulille":
        pass #TODO
    elif  args.dataset == "dlbclmorph":
        for fold in os.listdir(args.mDATA0_dir_train0):
            if fold != args.fold:
                for p in os.listdir(os.path.join(args.mDATA0_dir_train0, fold)):
                    y = labels[int(p)]
                    patches = glob.glob(os.path.join(args.mDATA0_dir_train0, fold, p, '*.pt'))
                    data.append((patches, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    elif  args.dataset == "bci":
        for img_name in os.listdir(os.path.join(args.mDATA0_dir_train0, "train")):
            y = img_name.split('_')[-1]
            patches = glob.glob(os.path.join(args.mDATA0_dir_train0, "train", img_name, '*.pt'))
            data.append((patches, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    
    n = len(data) - int(validation_factor * len(data))
    random.shuffle(data)
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, 
                                             epoch, criterion=None,  params=None, f_log=None, writer=None, 
                                             numGroup=3, total_instance=3, distill='MaxMinS'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(device=device)
    gt_0 = torch.LongTensor().to(device=device)
    gPred_1 = torch.FloatTensor().to(device=device)
    gt_1 = torch.LongTensor().to(device=device)

    with torch.no_grad():
        
        for patches, label in tqdm.tqdm(mDATA_list, ncols=50):

            tslideLabel = torch.LongTensor([label]).to(device=device)
            tfeat = torch.stack(list(map(torch.load, patches))).to(device=device)

            midFeat = dimReduction(tfeat)

            AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

            allSlide_pred_softmax = []

            for jj in range(params.num_MeanInference):

                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                slide_d_feat = []
                slide_sub_preds = []
                slide_sub_labels = []

                for tindex in index_chunk_list:
                    slide_sub_labels.append(tslideLabel)
                    idx_tensor = torch.LongTensor(tindex).to(device=device)
                    tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                    tAA = AA.index_select(dim=0, index=idx_tensor)
                    tAA = torch.softmax(tAA, dim=0)
                    tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                    tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                    slide_sub_preds.append(tPredict)

                    patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                    patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                    _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                    if distill == 'MaxMinS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'MaxS':
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx = topk_idx_max
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)
                    elif distill == 'AFS':
                        slide_d_feat.append(tattFeat_tensor)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                test_loss0.update(loss0.item(), numGroup)

                gSlidePred = UClassifier(slide_d_feat)
                allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

            allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
            allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
            gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
            gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

            loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
            test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    # gPred_0 = gPred_0[:, -1]
    # gPred_1 = gPred_1[:, -1]

    gt_0 = gt_0.to(device='cpu')
    gt_1 = gt_1.to(device='cpu')
    gPred_0 = gPred_0.to(device='cpu')
    gPred_1 = gPred_1.to(device='cpu')
    
    gt_0 = F.one_hot(gt_0, num_classes=len(TRAIN_PARAMS['class_to_label'][params.dataset])).numpy()
    gt_1 = F.one_hot(gt_1, num_classes=len(TRAIN_PARAMS['class_to_label'][params.dataset])).numpy()

    metrics_0 = metrics_fn(params, gt_0, gPred_0)
    metrics_1 = metrics_fn(params, gt_1, gPred_1)

    print_log('  First-Tier {}'.format(metrics_0), f_log)
    print_log('  Second-Tier {}'.format(metrics_1), f_log)

    writer.add_scalar(f'acc_0 ', metrics_0["accuracy"], epoch)
    writer.add_scalar(f'acc_1 ', metrics_1["accuracy"], epoch)

    return metrics_1["accuracy"]


def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, epoch, 
                                    ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    for patches, label in tqdm.tqdm(mDATA_list, ncols=50):

        tslideLabel = torch.LongTensor([label]).to(device=device)

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        tfeat_tensor = torch.stack(list(map(torch.load, patches))).to(device=device)

        feat_index = list(range(tfeat_tensor.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(device=device))
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)

        ## optimization for the second tier
        gSlidePred = UClassifier(slide_pseudo_feat)
        loss1 = ce_cri(gSlidePred, tslideLabel).mean()
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
        
        # AVOID GRADIENT ERROR !
        optimizer0.step()
        optimizer1.step()

        Train_Loss0.update(loss0.item(), numGroup)
        Train_Loss1.update(loss1.item(), 1)

    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))

    # model
    in_chn = 1024
    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(device=device)
    attention = Attention(params.mDim).to(device=device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(device=device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(device=device)

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)

    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

    model_path = os.path.join(params.output, params.name)
    os.makedirs(model_path, exist_ok=True)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log.txt')
    save_dir = os.path.join(model_path, 'best_model.pth')
    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')

    train_data, val_data = load_data(params)
    # val_data = load_data(params.mDATA0_dir_val0, params)

    print_log(f'training slides: {len(train_data)}, validation slides: {len(val_data)}', log_file)

    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr,  weight_decay=params.weight_decay)

    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    best_auc = 0
    best_epoch = -1

    for ii in range(params.EPOCH):

        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f' current learn rate {curLR}', log_file )

        train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=attCls, 
                                        mDATA_list=train_data, ce_cri=ce_cri, 
                                        optimizer0 = optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii, params=params, 
                                        f_log=log_file, writer=writer, numGroup=params.numGroup, total_instance=params.total_instance, 
                                        distill=params.distill_type)
        
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
        auc_val = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                           UClassifier=attCls, mDATA_list=val_data, 
                                                           criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, 
                                                           numGroup=params.numGroup_test, total_instance=params.total_instance_test, 
                                                           distill=params.distill_type)
        print_log(' ', log_file)

        if ii > int(params.EPOCH * 0.1):
            if auc_val > best_auc:
                best_auc = auc_val
                best_epoch = ii
                tsave_dict = {
                    'classifier': classifier.state_dict(),
                    'dim_reduction': dimReduction.state_dict(),
                    'attention': attention.state_dict(),
                    'att_classifier': attCls.state_dict()
                }
                torch.save(tsave_dict, save_dir)

            print_log(f' validation auc: {auc_val}, from epoch {best_epoch}', log_file)

        scheduler0.step()
        scheduler1.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mDATA0_dir_train0', required=True, type=str)  ## Train Set
    # parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
    # parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"])
    parser.add_argument('--output', type=str, required=True, help='where to store the model')
    parser.add_argument('--labels', type=str, default=None, help='path to labels CSV file')
    parser.add_argument('--fold', type=str, default=None, help='fold to use as test')
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--name', default='abc', type=str)
    parser.add_argument('--EPOCH', default=200, type=int)
    parser.add_argument('--epoch_step', default='[100]', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--isPar', default=False, type=bool)
    parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
    parser.add_argument('--train_show_freq', default=40, type=int)
    parser.add_argument('--droprate', default='0', type=float)
    parser.add_argument('--droprate_2', default='0', type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--batch_size_v', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_cls', default=2, type=int)
    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--total_instance', default=4, type=int)
    parser.add_argument('--numGroup_test', default=4, type=int)
    parser.add_argument('--total_instance_test', default=4, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--isSaveModel', action='store_false')
    parser.add_argument('--debug_DATA_dir', default='', type=str)
    parser.add_argument('--numLayer_Res', default=0, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--num_MeanInference', default=1, type=int)
    parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
    params = parser.parse_args()

    global TRAIN_PARAMS
    TRAIN_PARAMS = dict(
        # dictionnar to convert class name to label
        class_to_label = {
            "chulille": {'ABC': 1, 'GCB': 0},
            "dlbclmorph": {'NGC': 1, 'GC': 0},
            "bci": {'0': 0, '1+': 1, '2+': 2, '3+': 3},
        },
    )

    params.num_cls = len(TRAIN_PARAMS["class_to_label"][params.dataset])

    if params.dataset in ["chulille", "dlbclmorph"]:
        assert not params.labels is None, "path to labels file must be provided for datasets [chulille, dlbclmorph]"

    if params.dataset == "dlbclmorph":
        assert not params.fold is None, "for dataset dlbclmorph, fold id must be provided"

    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    np.random.seed(32)
    random.seed(32)

    main(params)
