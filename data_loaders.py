import os, glob, random, math
import numpy as np
import pandas

def select_samples(samples, n):
    """
    Select samples such that all patients contribute
    
    Args: 
        samples : list of list of dict path to slides fragments
                [
                    patient 1
                    [   {'HES': path/to/sequence, 'BCL6': path/to/sequence, ...},
                        {'HES': path/to/sequence, 'BCL6': path/to/sequence, ...},
                        ...],
                    patient 2
                    [   {'HES': path/to/sequence, 'BCL6': path/to/sequence, ...},
                        {'HES': path/to/sequence, 'BCL6': path/to/sequence, ...},
                        ...],
                    ... 
                ]
        n : number of total samples (int)

    Return:
        data : dataset of length at most n (list)
    """
    
    # make sure that all patient contribute to the set
    data = []
    while len(data) <= n:
        for patient in samples:
            try:
                data.append(patient.pop())
            except IndexError:
                continue

        # check if all samples used
        if not any(samples):
            break
    
    if math.isinf(n):
        return data
    
    return data[:n]

class CHULille:
    def __init__(self, args, class2label, split: str, list_of_folds=None, *useless_args):
        self.class2label = class2label
        self.args = args
        self.split = split
        self.list_of_folds = list_of_folds
        
        # load labels
        self.labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

        # load data
        if not list_of_folds is None:
            # load heterogeneous only for train data
            load_heterogeneous = True if (split == 'train' and args.heterogeneous_batch) else False
            self.data, self.total = self.load_folds(args.data, list_of_folds, load_heterogeneous)
        elif split == 'train' and args.heterogeneous_batch:
            self.data, self.total = self.load_heterogeneous(os.path.join(args.data, "train"))
        elif split == 'train':
            self.data, self.total = self.load_standard(os.path.join(args.data, "train"))
        else:
            self.data, self.total = self.load_standard(os.path.join(args.data, "test"))

        # initialize iteration counter
        self.iter = -1

    def load_patient(self, patient_path):
        """
            Load samples of a patient.
            Returns a list like:
            [
                {'stain 1': path/to/seq, 'stain 2': path/to/seq, ...},
                {'stain 1': path/to/seq, 'stain 2': path/to/seq, ...},
                ...
            ]
        """
        patient_samples = []
        for object_id in os.listdir(patient_path):
            samples = {}
            for stain in os.listdir(os.path.join(patient_path, object_id)):
                samples.update({stain: glob.glob(os.path.join(patient_path, object_id, stain, "*.npy"))})
            
            # shuffl stains samples
            for k in samples.keys():
                random.shuffle(samples[k])
            
            # creates list of samples with stains grouped
            grouped_samples = []
            while all(samples.values()):   # check all stain have at least one patch in each split
                one_sample = {stain: seq.pop() for stain, seq in samples.items()}
                grouped_samples.append(one_sample)

            patient_samples.extend(grouped_samples)

        return patient_samples
        
    def load_folds(self, path, list_of_folds, load_heterogeneous):
        if load_heterogeneous:
            data = dict()
            for fold in list_of_folds:
                fold_data, _ = self.load_heterogeneous(os.path.join(path, fold))
                data.update(fold_data)
        
            total = [self.labels[int(patient)] for patient in data.keys()]
            total = {k: total.count(k) for k in set(total)}
            return data, total
        else:
            dict_of_classes_samples = {k: [] for k in self.class2label.keys()}
            samples_per_class = {k: 0 for k in self.class2label.keys()}
            
            for fold in list_of_folds:
                for patient in os.listdir(os.path.join(path, fold)):
                    label = self.labels[int(patient)]
                    patient_samples = self.load_patient(os.path.join(path, fold, patient))
                    dict_of_classes_samples[label].append(patient_samples)
                    samples_per_class[label] += len(patient_samples)

            # get minimum between classes number of samples
            n = min(samples_per_class.values())

            data = []
            total = {}
            for label, samples in dict_of_classes_samples.items():

                if self.args.no_class_balance:
                    class_samples = select_samples(samples, float('inf'))
                else:
                    class_samples = select_samples(samples, n)

                # append label index to each sample
                y = np.full(len(class_samples), self.class2label[label])
                class_samples = list(zip(class_samples, y))
                
                # add to data list and update class counter
                data.extend(class_samples)
                total[label] = len(class_samples)

            return data, total

    def load_heterogeneous(self, path):
        data = dict()
        for patient in os.listdir(path):
            patient_samples = self.load_patient(os.path.join(path, patient))
            data.update({patient : patient_samples})
            
        total = [self.labels[int(patient)] for patient in data.keys()]
        total = {k: total.count(k) for k in set(total)}
        return data, total
    
    def load_standard(self, path):
        dict_of_classes_samples = {k: [] for k in self.class2label.keys()}
        samples_per_class = {k: 0 for k in self.class2label.keys()}
        
        for patient in os.listdir(path):
            label = self.labels[int(patient)]
            patient_samples = self.load_patient(os.path.join(path, patient))
            dict_of_classes_samples[label].append(patient_samples)
            samples_per_class[label] += len(patient_samples)

        # get minimum between classes number of samples
        n = min(samples_per_class.values())

        data = []
        total = {}
        for label, samples in dict_of_classes_samples.items():

            if self.args.no_class_balance:
                class_samples = select_samples(samples, float('inf'))
            else:
                class_samples = select_samples(samples, n)

            # append label index to each sample
            y = np.full(len(class_samples), self.class2label[label])
            class_samples = list(zip(class_samples, y))
            
            # add to data list and update class counter
            data.extend(class_samples)
            total[label] = len(class_samples)

        return data, total
    
    def __len__(self):
        # return the number of batches
        if isinstance(self.data, dict):
            return math.ceil(len(self.data.keys()) / self.args.batch_size)
        else:
            return math.ceil(len(self.data) / self.args.batch_size)
    
    def __iter__(self):
        # shuffle data
        if isinstance(self.data, dict):
            self.data = {k: self.data[k] for k in random.sample(list(self.data.keys()), k=len(self.data.keys()))}
        else:
            random.shuffle(self.data)
        
        # we start at -batch_size because iter is being updated at each next iteration before accessing data
        self.iter = -self.args.batch_size
        
        return self
    
    def __next__(self):

        def read_data(x, y):
            # read stains sequences
            read_sequences = lambda i : {k: np.load(i[k]) for k in i.keys()}
            x = list(map(read_sequences, x))

            # create ground-truth
            if isinstance(self.data, dict):
                get_patient_class = lambda p : int(self.class2label(self.labels[int(p)]))
                y = map(get_patient_class, y)
            y = list(y)
            
            return x, y

        self.iter += self.args.batch_size

        # return next batch if iter is lower than dataset size
        if isinstance(self.data, dict):
            if self.iter < len(self.data.keys()):
                batch = [(random.choice(self.data[k]), k) for k in list(self.data.keys())[self.iter:self.iter + self.args.batch_size]]
                x, y = zip(*batch)
                return read_data(x, y)
        else:
            if self.iter < len(self.data):
                batch = self.data[self.iter:self.iter + self.args.batch_size]
                x, y = zip(*batch)
                return read_data(x, y)
        
        # reinitialise iter and return StopIteration when run out of data
        self.iter = -self.args.batch_size
        return StopIteration


def load_patient(patient_path):
    patient_samples = {}
    for stain in os.listdir(os.path.join(patient_path)):
        patient_samples.update({stain: glob.glob(os.path.join(patient_path, stain, "*.pt"))})
    
    # shuffl stains samples
    for k in patient_samples.keys():
        random.shuffle(patient_samples[k])

    return patient_samples
        
def load_folds(args, list_of_folds, class2label, labels):
    data = []
    for fold in list_of_folds:
        for patient in os.listdir(os.path.join(args.data, fold)):
            label = labels[int(patient)]
            patient_samples = load_patient(os.path.join(args.data, fold, patient))
            data.append((patient_samples, label))

    total = [i[1] for i in data]
    total = {k: total.count(k) for k in set(total)}
    data = [(i[0], class2label[i[1]]) for i in data]
    return data, total

def DLBCLMorph(args, class2label, list_of_folds, validation_factor, *useless_args):

    # load labels
    labels = pandas.read_csv(args.labels).set_index('patient_id')['label'].to_dict()

    # load data
    data, _ = load_folds(args, list_of_folds, class2label, labels)
    n = len(data) - int(len(data)*validation_factor)
    random.shuffle(data)
    train_data, val_data = data[:n], data[n:]

    # compute classes distribution
    train_total = [k for k, v in class2label.items() for i in train_data if v == i[1]]
    train_total = {k: train_total.count(k) for k in set(train_total)}

    val_total = [k for k, v in class2label.items() for i in val_data if v == i[1]]
    val_total = {k: val_total.count(k) for k in set(val_total)}

    return (train_data, train_total), (val_data, val_total)

def BCI(args, class2label, validation_factor, *useless_args):
    data = []
    for img in os.listdir(os.path.join(args.data, 'train')):
        img_samples = load_patient(os.path.join(os.path.join(args.data, 'train'), img))
        y = img.split('_')[-1]
        y = class2label[y]
        data.append((img_samples, y))
    
    # split data
    n = len(data) - int(len(data)*validation_factor)
    random.shuffle(data)
    train_data, val_data = data[:n], data[n:]
    
    # compute classes distribution
    train_total = [k for k, v in class2label.items() for i in train_data if v == i[1]]
    train_total = {k: train_total.count(k) for k in set(train_total)}

    val_total = [k for k, v in class2label.items() for i in val_data if v == i[1]]
    val_total = {k: val_total.count(k) for k in set(val_total)}

    return (train_data, train_total), (val_data, val_total)
