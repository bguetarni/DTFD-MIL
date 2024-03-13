import argparse, os, glob, random, re
import pandas
import numpy as np
import tqdm
from sklearn import metrics as sklearn_metrics

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from Model.Attention import Attention_with_Classifier, Attention_Gated
from Model.network import    Classifier_1fc, DimReduction

def metrics_fn(args, Y_true, Y_pred):
    """
    args : script arguments
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    """

    # cross-entropy error
    error = sklearn_metrics.log_loss(Y_true, Y_pred)

    # ROC AUC (per class)
    auc = dict()
    for i in range(Y_true.shape[1]):
        # select class one-hot values
        ytrue = Y_true[:,i]

        # transform probabilities from [0.5,1] to [0,1]
        # probabilities in [0,0.5] are clipped to 0
        ypred = np.clip(Y_pred[:,i], 0.5, 1) * 2 - 1
        auc_score = sklearn_metrics.roc_auc_score(ytrue, ypred)
        auc.update({i: auc_score})
    
    # convert to one-hot index
    Y_true_label = np.argmax(Y_true, axis=-1)
    Y_pred_label = np.argmax(Y_pred, axis=-1)
    
    # global metrics
    TP = sklearn_metrics.accuracy_score(Y_true_label, Y_pred_label, normalize=False)
    accuracy = sklearn_metrics.accuracy_score(Y_true_label, Y_pred_label)
    micro_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='micro')
    macro_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='macro')
    weighted_Fscore = sklearn_metrics.f1_score(Y_true_label, Y_pred_label, average='weighted')

    # compile metrics in dict
    metrics_ = dict(error=error, TP=TP, accuracy=accuracy, micro_Fscore=micro_Fscore, macro_Fscore=macro_Fscore, weighted_Fscore=weighted_Fscore)
    
    # confusion matrix for each class
    multiclass_cm = sklearn_metrics.multilabel_confusion_matrix(Y_true_label, Y_pred_label)

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


def load_data(args):

    if args.dataset in ["chulille", "dlbclmorph"]:
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    data = []
    if args.dataset == "chulille":
        print('WARNING: for dataset chulille, only HES stain available yet !')
        for fold in os.listdir(args.mDATA_dir_test0):
            if fold == args.fold:
                for slide in os.listdir(os.path.join(args.mDATA_dir_test0, fold)):
                    y = labels[int(re.findall('\d+', slide)[0])]
                    patches = {stain: glob.glob(os.path.join(args.mDATA_dir_test0, fold, slide, '*.pt')) for stain in TRAIN_PARAMS['STAINS'][args.dataset][:1]}
                    data.append((patches, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    elif  args.dataset == "dlbclmorph":
        for fold in os.listdir(args.mDATA_dir_test0):
            if fold == args.fold:
                for p in os.listdir(os.path.join(args.mDATA_dir_test0, fold)):
                    y = labels[int(p)]
                    patches = {stain: glob.glob(os.path.join(args.mDATA_dir_test0, fold, p, stain, '*.pt')) for stain in TRAIN_PARAMS['STAINS'][args.dataset]}
                    data.append((patches, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    elif  args.dataset == "bci":
        for img_name in os.listdir(os.path.join(args.mDATA_dir_test0, "test")):
            y = img_name.split('_')[-1]
            patches = {stain: glob.glob(os.path.join(args.mDATA_dir_test0, "test", img_name, stain, '*.pt')) for stain in TRAIN_PARAMS['STAINS'][args.dataset]}
            data.append((patches, TRAIN_PARAMS['class_to_label'][args.dataset][y]))
    
    return data


def main(args):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model name
    model_name = os.path.split(args.model_dir)[1]
    print('testing model {} on dataset {}'.format(model_name, args.dataset))
    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # model
    in_chn = 1024
    classifier = Classifier_1fc(args.mDim, args.num_cls, args.droprate)
    attention = Attention_Gated(args.mDim)
    dimReduction = DimReduction(in_chn, args.mDim, numLayer_Res=args.numLayer_Res)
    attCls = Attention_with_Classifier(stains=None, L=args.mDim, num_cls=args.num_cls, droprate=args.droprate_2)
    model = dict(classifier=classifier, attention=attention, dimReduction=dimReduction, attCls=attCls)
    model = torch.nn.ModuleDict(model).eval().to(device=device)
    state_dict = torch.load(os.path.join(args.model_dir, "ckpt.pth"), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)

    instance_per_group = args.total_instance // args.numGroup

    data = load_data(args)

    result_per_patient_or_image = []
    for patches, label in tqdm.tqdm(data, ncols=50):
        patches = patches[TRAIN_PARAMS['STAINS'][args.dataset][0]]
        tfeat = torch.stack(list(map(torch.load, patches))).to(device=device)
            
        with torch.no_grad():

            midFeat = model["dimReduction"](tfeat)
            AA = model["attention"](midFeat, isNorm=False).squeeze(0)  ## N

            allSlide_pred_softmax = []
            for jj in range(args.num_MeanInference):

                feat_index = list(range(tfeat.shape[0]))
                random.shuffle(feat_index)
                index_chunk_list = np.array_split(np.array(feat_index), args.numGroup)
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

                    # if args.distill_type == 'MaxMinS':
                    #     topk_idx_max = sort_idx[:instance_per_group].long()
                    #     topk_idx_min = sort_idx[-instance_per_group:].long()
                    #     topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    #     d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    #     slide_d_feat.append(d_inst_feat)
                    # elif args.distill_type == 'MaxS':
                    #     topk_idx_max = sort_idx[:instance_per_group].long()
                    #     topk_idx = topk_idx_max
                    #     d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                    #     slide_d_feat.append(d_inst_feat)
                    # elif args.distill_type == 'AFS':
                    #     slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat.append(tattFeat_tensor)

                slide_d_feat = torch.cat(slide_d_feat, dim=0)
                gSlidePred = model["attCls"](slide_d_feat)
                allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

        allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
        y_pred = torch.mean(allSlide_pred_softmax, dim=0).to('cpu').numpy()
        
        # duplicate patient label to all samples
        y_true = np.zeros(len(TRAIN_PARAMS['class_to_label'][args.dataset]), dtype='int32')
        y_true[label] = 1
        
        # add to dict
        result_per_patient_or_image.append((y_true, y_pred))
    
    # calculates metrics across patches
    y_true, y_pred = zip(*result_per_patient_or_image)
    y_true = np.stack(y_true)
    y_pred = np.stack(y_pred)
    patches_metrics = metrics_fn(args, y_true, y_pred)
    
    df = pandas.DataFrame([patches_metrics])
    
    # round floats to 2 decimals
    df = df.round(decimals=2)

    # save results in a CSV file
    df.to_csv(os.path.join(output_dir, "metrics.csv"), sep=';')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"])
    parser.add_argument('--mDATA_dir_test0', required=True, type=str)         ## Test Set
    parser.add_argument('--model_dir', required=True, type=str, help='path to model weights directory')
    parser.add_argument('--labels', default=None, type=str)
    parser.add_argument('--fold', type=str, default=None, help='fold to use as test')
    parser.add_argument('--output', required=True, type=str, help='path to the CSV file to save results')
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--num_cls', required=True, type=int)

    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--total_instance', default=4, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    parser.add_argument('--numLayer_Res', default=0, type=int)
    parser.add_argument('--droprate', default='0', type=float)
    parser.add_argument('--droprate_2', default='0', type=float)
    parser.add_argument('--num_MeanInference', default=1, type=int)
    parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
    args = parser.parse_args()

    global TRAIN_PARAMS
    TRAIN_PARAMS = dict(

        # stains of each dataset
        STAINS = {
            "chulille": ['HES', 'BCL6', 'CD10', 'MUM1'],
            "dlbclmorph": ['HE', 'BCL6', 'CD10', 'MUM1'],
            "bci": ['HES', 'IHC'],
        },

        # dictionnar to convert class name to label
        class_to_label = {
            "chulille": {'ABC': 1, 'GCB': 0},
            "dlbclmorph": {'NGC': 1, 'GC': 0},
            "bci": {'0': 0, '1+': 1, '2+': 2, '3+': 3},
        },
    )

    args.num_cls = len(TRAIN_PARAMS["class_to_label"][args.dataset])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(32)
    torch.cuda.manual_seed(32)
    np.random.seed(32)
    random.seed(32)

    main(args)
