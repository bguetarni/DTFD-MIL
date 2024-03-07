import os, math, random, json, argparse, itertools, glob, pickle
import tqdm
import numpy as np
import pandas
import torch, torch.nn as nn
import einops
from sklearn import metrics
from PIL import Image

from Model import DTFDMIL
import utils, data_loaders

def metrics_fn(args, Y_true, Y_pred):
    """
    args : script arguments
    Y_true (n_samples, n_classes) : array of probabilities prediction
    Y_pred (n_samples, n_classes) : array of true class as one-hot index
    """

    # cross-entropy error
    error = metrics.log_loss(Y_true, Y_pred)

    # ROC AUC (per class)
    auc = dict()
    for i in range(Y_true.shape[1]):
        # select class one-hot values
        ytrue = Y_true[:,i]

        # transform probabilities from [0.5,1] to [0,1]
        # probabilities in [0,0.5] are clipped to 0
        ypred = np.clip(Y_pred[:,i], 0.5, 1) * 2 - 1
        auc_score = metrics.roc_auc_score(ytrue, ypred)
        auc.update({i: auc_score})
    
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
    for k, i in TEST_PARAMS['class_to_label'][args.dataset].items():
        
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

def load_data(args, TEST_PARAMS):
    if args.dataset in ["chulille", "dlbclmorph"]:
        # load labels
        labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

        data = dict()
        for patient in tqdm.tqdm(os.listdir(args.data), ncols=50):
            if args.dataset == "chulille":
                raise RuntimeError("dataset chulille not to be used for now")
            else:
                patient_samples = data_loaders.load_patient(os.path.join(args.data, patient))
            
            # select only HE/HES data
            patient_samples = patient_samples[TEST_PARAMS['STAINS'][args.dataset][0]]
            
            # sample MAX_SAMPLE_PER_PATIENT examples per patient
            if len(patient_samples) > TEST_PARAMS['MAX_SAMPLE_PER_PATIENT']:
                patient_samples = random.sample(patient_samples, TEST_PARAMS['MAX_SAMPLE_PER_PATIENT'])
            
            # duplicate patient label to all samples
            true_idx = TEST_PARAMS['class_to_label'][args.dataset][labels[int(patient)]]
            y = np.zeros(len(TEST_PARAMS['class_to_label'][args.dataset]), dtype='int32')
            y[true_idx] = 1
            
            # add to data
            data.update({patient: (patient_samples, y)})
    elif args.dataset == "bci":
        data = []
        for img_name in os.listdir(os.path.join(args.data, 'test')):
            x = glob.glob(os.path.join(args.data, 'test', img_name, TEST_PARAMS['STAINS'][args.dataset][0], "*.pt"))
            true_idx = TEST_PARAMS['class_to_label'][args.dataset][img_name.split('_')[-1]]
            y = np.zeros(len(TEST_PARAMS['class_to_label'][args.dataset]), dtype='int32')
            y[true_idx] = 1
            data.append((img_name, (x, y)))
    
    return data

def model_predict_on_list(model, args, data):

    y_pred = []
    y_true = []
    for batch in data:
        # prepare batch
        x, y = batch

        # load input and group into bags
        x = torch.stack(list(map(torch.load, x)), dim=0).to(device=device)
        x = utils.split_into_bags(x, args.numGroup)
        
        # inference
        with torch.no_grad():
            slide_pred, sub_bags_pred = model(x)

        y_pred.append(slide_pred.to(device='cpu').numpy())
        y_true.append(y)

    # all prediction in one array
    y_pred = np.concatenate(y_pred)
    y_true = np.stack(y_true, axis=0)
    
    return y_pred, y_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"], help='dataset')    
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--model', type=str, required=True, help='path to the model directory')
    parser.add_argument('--output', type=str, required=True, help='path to the directory to save results')
    parser.add_argument('--labels', type=str, default=None, help='path to csv file with labels')
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use (e.g. 0)')

    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--in_chn', default=1024, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    args = parser.parse_args()

    if args.dataset == "chulille":
        num_classes = 2
    elif args.dataset == "dlbclmorph":
        num_classes = 2
    elif args.dataset == "bci":
        num_classes = 4
    
    global TEST_PARAMS
    TEST_PARAMS = dict(

        # how many sample per patient to use
        MAX_SAMPLE_PER_PATIENT = 10000,
        
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

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    print('device: ', device)

    # get model name
    model_name = os.path.split(args.model)[1]
    print('testing model {} on dataset {}'.format(model_name, args.dataset))
    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = DTFDMIL.Student(in_channels=args.in_chn, mDim=args.mDim, num_cls=num_classes)
    
    # load model weights
    print('load model weights..')
    model_weights = os.path.join(args.model, "ckpt.pth")
    state_dict = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # multi GPU
    model = nn.DataParallel(model).eval().to(device=device)

    print('loading data..')
    data = load_data(args, TEST_PARAMS)
    
    if isinstance(data, dict):
        print('predict per patient')
    
        patient, patient_data = zip(*data.items())
        patient_data = list(patient_data)

        # model inference
        y_pred, y_true = model_predict_on_list(model, args, patient_data)

        # save images prediction for ROC curve
        result_per_patient = {p: (ytrue, ypred) for p, ytrue, ypred in zip(patient, y_true, y_pred)}
        with open(os.path.join(output_dir, "patient_prediction.pickle"), "wb") as f:
            pickle.dump(result_per_patient, f)

        # calculates metrics on patients
        y_true, y_pred = zip(*result_per_patient.values())
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        metrics = metrics_fn(args, y_true, y_pred)

        # build model DataFrame with results
        df = pandas.DataFrame({model_name: metrics.values()}, index=metrics.keys()).transpose()
    else:
        image_names, data = zip(*data)
        data = list(data)
        
        # model inference
        print('predicting on all data..')
        y_pred, y_true = model_predict_on_list(model, args, data)

        # save images prediction for ROC curve
        result_per_image = {name: (ytrue, ypred) for name, ytrue, ypred in zip(image_names, y_true, y_pred)}
        with open(os.path.join(output_dir, "images_prediction.pickle"), "wb") as f:
            pickle.dump(result_per_image, f)

        # calculates metrics
        metrics = metrics_fn(args, y_true, y_pred)

        # build model DataFrame with results
        df = pandas.DataFrame({model_name: metrics.values()}, index=metrics.keys()).transpose()

    # round floats to 2 decimals
    df = df.round(decimals=2)

    # save results in a CSV file
    df.to_csv(os.path.join(output_dir, "metrics.csv"), sep=';')
