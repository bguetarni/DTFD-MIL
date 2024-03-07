import os, pickle, time, argparse, json, random
import tqdm
import numpy as np
import torch, torch.nn as nn

from Model import DTFDMIL
import utils, data_loaders

def StudentLoss(pred_student, pred_teacher):
    """
    Loss function for knowledge distillation (handle soft and hard distillation)
    
    Args:
        pred_student : student output
        pred_teacher : teacher output
    """

    # logit distillation loss
    if TRAIN_PARAMS['DISTIL_HARD']:
        hard_labels = pred_teacher.argmax(dim=-1)
        distil_loss = utils.smooth_cross_entropy(pred_student, hard_labels, args.smooth_label)
    else:
        softened_softmax = lambda x : nn.functional.softmax(x / TRAIN_PARAMS['DISTIL_SOFTMAX_TEMP'], dim=-1)
        distil_loss = nn.functional.kl_div(torch.log(softened_softmax(pred_student)), softened_softmax(pred_teacher), reduction='batchmean')
        distil_loss = (TRAIN_PARAMS['DISTIL_SOFTMAX_TEMP'] ** 2) * distil_loss

    return distil_loss


def validation(args, model, validation_data):
    """
    Validation stage
    
    Args:
        args : arguments
        model : model to train
        validation_data : iterable that contains validation data that is already splitted in batches

    Return:
        loss : total loss
    """
    
    Y_pred_slide, Y_slide = [], []
    for batch in tqdm.tqdm(validation_data, ncols=50):
        if batch is StopIteration:
            break
        
        x, y = batch
        
        # load input and group into bags
        x = {k: torch.stack(list(map(torch.load, v)), dim=0).to(device=device) for k, v in x.items()}
        x = utils.split_into_bags(x, args.numGroup)
        
        with torch.no_grad():
            if args.stain == "mono":
                # select first stain for each bag
                x = [i[TRAIN_PARAMS['STAINS'][args.dataset][0]] for i in x]
            
            slide_pred, _ = model(x)

        Y_pred_slide.append(slide_pred.to(device='cpu'))
        Y_slide.append(y)

    # concatenate batches predictions and labels
    Y_pred_slide = torch.cat(Y_pred_slide, dim=0)
    Y_slide = torch.tensor(Y_slide)
    
    # computes loss
    loss = nn.functional.cross_entropy(Y_pred_slide, Y_slide)
    
    return loss.item()

def train_step(args, batch, model, teacher=None):
    """
    Train step for a single batch
    
    Args:
        args : arguments
        batch : a single batch
        model : model to train
        teacher : teacher model if training student

    Return:
        batch_loss : loss on batch
    """

    x, y = batch

    # load input and group into bags
    x = {k: torch.stack(list(map(torch.load, v)), dim=0).to(device=device) for k, v in x.items()}
    x = utils.split_into_bags(x, args.numGroup)

    # forward
    if teacher is None:
        if args.stain == 'mono':
            # select first stain for each bag
            x = [i[TRAIN_PARAMS['STAINS'][args.dataset][0]] for i in x]
        
        slide_pred, sub_bags_pred = model(x)

        # tier 1 loss
        sub_bags_targets = torch.full((sub_bags_pred.shape[0],), y, dtype=torch.long).to(device=device)
        sub_bags_loss = nn.functional.cross_entropy(sub_bags_pred, sub_bags_targets)
        optimizer_adam0.zero_grad()
        sub_bags_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.module.dimReduction.parameters(), args.grad_clipping)
        torch.nn.utils.clip_grad_norm_(model.module.tier1.parameters(), args.grad_clipping)

        # tier 2 loss
        slide_targets = torch.full((slide_pred.shape[0],), y, dtype=torch.long).to(device=device)
        slide_loss = nn.functional.cross_entropy(slide_pred, slide_targets)
        optimizer_adam1.zero_grad()
        slide_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.tier2.parameters(), args.grad_clipping)
    else:
        # select first stain for each bag
        student_x = [i[TRAIN_PARAMS['STAINS'][args.dataset][0]] for i in x]

        # student forward
        slide_pred_student, sub_bags_pred_student = model(student_x)

        # teacher forward
        with torch.no_grad():
            slide_pred_teacher, sub_bags_pred_teacher = teacher(x)

        # KD loss
        pred_student = torch.cat((slide_pred_student, sub_bags_pred_student), dim=0)
        pred_teacher = torch.cat((slide_pred_teacher, sub_bags_pred_teacher), dim=0)
        distil_loss = StudentLoss(pred_student, pred_teacher)
        
        # tier 1 loss
        sub_bags_targets = torch.full((sub_bags_pred_student.shape[0],), y, dtype=torch.long).to(device=device)
        sub_bags_loss = nn.functional.cross_entropy(sub_bags_pred_student, sub_bags_targets)
        sub_bags_loss = TRAIN_PARAMS['DISTIL_TRUE_WEIGHT'] * sub_bags_loss + TRAIN_PARAMS['DISTIL_TEACHER_WEIGHT'] * distil_loss
        optimizer_adam0.zero_grad()
        sub_bags_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.module.dimReduction.parameters(), args.grad_clipping)
        torch.nn.utils.clip_grad_norm_(model.module.tier1.parameters(), args.grad_clipping)

        # tier 2 loss
        slide_targets = torch.full((slide_pred_student.shape[0],), y, dtype=torch.long).to(device=device)
        slide_loss = nn.functional.cross_entropy(slide_pred_student, slide_targets)
        slide_loss = TRAIN_PARAMS['DISTIL_TRUE_WEIGHT'] * slide_loss + TRAIN_PARAMS['DISTIL_TEACHER_WEIGHT'] * distil_loss
        optimizer_adam1.zero_grad()
        slide_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.tier2.parameters(), args.grad_clipping)
    
    # update parameters
    optimizer_adam0.step()
    optimizer_adam1.step()

    # total batch loss
    batch_loss = slide_loss + sub_bags_loss
    
    return batch_loss.detach().to(device='cpu').item()

if __name__ == '__main__':
    
    # general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["chulille", "dlbclmorph", "bci"], help='dataset')    
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--output', type=str, required=True, help='path to folder to save model weights and metadata')
    parser.add_argument('--labels', type=str, default=None, help='path to csv file with labels')
    parser.add_argument('--stain', type=str, required=True, choices=["multi", "mono"], help='multi or mono-stain training')
    parser.add_argument('--teacher', type=str, default=None, help='path to the teacher if training the student')
    parser.add_argument('--name', type=str, required=True, help='name of the model')
    parser.add_argument('--description', type=str, default='', help='description of the model and train strategy')
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use (e.g. 0,1)')
    
    # options for some datasets
    parser.add_argument('--fold', type=str, default=None, help='name of fold to use as validation')
    parser.add_argument('--validation_factor', type=float, default=0.2, help='factor to isolate valdiation data from train data')
    
    # training arguments
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--smooth_label', type=float, default=0.1, help='label smoothing value')

    # DTFD-MIL arguments
    parser.add_argument('--epoch_step', default='[100]', type=str)
    parser.add_argument('--droprate', default='0', type=float)
    parser.add_argument('--droprate_2', default='0', type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--total_instance', default=4, type=int)
    parser.add_argument('--numGroup_test', default=4, type=int)
    parser.add_argument('--total_instance_test', default=4, type=int)
    parser.add_argument('--in_chn', default=1024, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--numLayer_Res', default=0, type=int)
    parser.add_argument('--num_MeanInference', default=1, type=int)
    parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
    args = parser.parse_args()

    if args.stain == "multi":
        print("training mul-stain teacher model: {}".format(args.name))
    elif not args.teacher is None:
        print("training mono-stain student model: {}".format(args.name))
    else:
        print("training mono-stain model without teacher: {}".format(args.name))

    if args.dataset == "chulille":
        num_classes = 2
    elif args.dataset == "dlbclmorph":
        num_classes = 2
    elif args.dataset == "bci":
        num_classes = 4
    
    global TRAIN_PARAMS
    TRAIN_PARAMS = dict(

        # distillation parameters
        DISTIL_HARD = True,
        DISTIL_SOFTMAX_TEMP = 3.0,
        DISTIL_TRUE_WEIGHT = 0.5,
        DISTIL_TEACHER_WEIGHT = 0.5,

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
    
    # create model folder
    model_path = os.path.join(args.output, args.name)
    os.makedirs(model_path, exist_ok=True)

    # build model and load base parameters
    if args.stain == "multi":
        model = DTFDMIL.Teacher(stains=TRAIN_PARAMS['STAINS'][args.dataset], 
                                in_channels=args.in_chn, mDim=args.mDim, num_cls=num_classes)
    else:
        model = DTFDMIL.Student(in_channels=args.in_chn, mDim=args.mDim, num_cls=num_classes)
        
        if not args.teacher is None:
            # load teacher model
            teacher = DTFDMIL.Teacher(stains=TRAIN_PARAMS['STAINS'][args.dataset], 
                                      in_channels=args.in_chn, mDim=args.mDim, num_cls=num_classes)
            state_dict = torch.load(os.path.join(args.teacher, "ckpt.pth"), map_location=torch.device('cpu'))
            teacher.load_state_dict(state_dict, strict=False)
    
    # multi GPU
    model = nn.DataParallel(model).to(device=device)
    if not args.teacher is None:
        teacher = nn.DataParallel(teacher).eval().to(device=device)

    # optimizer and learning rate scheduler
    trainable_parameters = []
    trainable_parameters += list(model.module.dimReduction.parameters())
    trainable_parameters += list(model.module.tier1.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=args.lr,  weight_decay=args.weight_decay)
    optimizer_adam1 = torch.optim.Adam(model.module.tier2.parameters(), lr=args.lr,  weight_decay=args.weight_decay)

    epoch_step = json.loads(args.epoch_step)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=args.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=args.lr_decay_ratio)
    
    with open(os.path.join(model_path, 'log'), 'w') as logger:

        # log the date and time
        logger.write(time.strftime("%x %X"))

        # log the training parameters
        logger.writelines(['\n{} : {}'.format(k,v) for k, v in TRAIN_PARAMS.items()])

        # log the training parameters
        logger.writelines(['\n{} : {}'.format(k,v) for k, v in vars(args).items()])

        # log the model size
        msg = ['\n\nmodel paramaters detail :\n']
        for n, submodule in model.module.named_children():
            msg.append('------ {}: {}\n'.format(n, utils.model_parameters_count(submodule)))

        msg.append('total : {}\n'.format(utils.model_parameters_count(model)))
        logger.writelines(msg)
    
    if not args.fold is None:
        list_of_folds = [i for i in os.listdir(args.data) if 'fold' in i]
        train_folds = [i for i in list_of_folds if i != args.fold]
        val_folds = [i for i in list_of_folds if i == args.fold]
    
    ###################   DATA LOADING   ##################
    if args.dataset == "chulille":
        raise RuntimeError("chulille may not be used for now")
    elif args.dataset == "dlbclmorph":
        (train_data, train_total), (test_data, test_total) = data_loaders.DLBCLMorph(args, TRAIN_PARAMS['class_to_label'][args.dataset], train_folds, args.validation_factor)
    elif args.dataset == "bci":
        (train_data, train_total), (test_data, test_total) = data_loaders.BCI(args, TRAIN_PARAMS['class_to_label'][args.dataset], args.validation_factor)

    print('\n train data :', train_total)
    with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log train data total
            logger.write('\ntrain data : {}'.format(train_total))

    print('\n validation data :', test_total)
    with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log train data total
            logger.write('\nvalidation data : {}'.format(test_total))
    ########################################################

    train_loss = []
    val_loss = []
    for epoch in range(args.n_epoch):
        print('\nepoch ', epoch+1)

        random.shuffle(train_data)

        print('train')
        model.train()
        loss = []
        for batch in tqdm.tqdm(train_data, ncols=50):
            if batch is StopIteration:
                break
                
            # train step
            if args.teacher is None:
                batch_loss = train_step(args, batch, model)
            else:
                batch_loss = train_step(args, batch, model, teacher)
            loss.append(batch_loss)
            
        train_loss.append(np.mean(loss))
        
        print('validation')
        model.eval()
        loss = validation(args, model, test_data)
        
        # learning rate schedule
        scheduler0.step()
        scheduler1.step()
        
        # append validation metrics
        val_loss.append(loss)
        
        # save the weights of the model if current validation error is lowest
        if val_loss[-1] == min(val_loss):
            torch.save(model.module.state_dict(), os.path.join(model_path, 'ckpt.pth'))
        
        # save results
        results = {'train loss' : train_loss, 'validation loss': val_loss}
        with open(os.path.join(model_path, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)

        with open(os.path.join(model_path, 'log'), 'a') as logger:
            # log the date and time of end of epoch
            logger.write('\nend of epoch {} : {}'.format(epoch+1, time.strftime("%x %X")))
