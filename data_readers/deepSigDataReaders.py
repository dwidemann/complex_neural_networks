import os
import pickle
import random
import numpy as np
import h5py
import random
from sklearn.preprocessing import StandardScaler
from functools import partial
from time import time
import os, wget
import tarfile
from sklearn.datasets import make_classification
from tqdm import tqdm

import h5py
import torch
from sklearn.preprocessing import StandardScaler
import pickle

def get_data_reader(dataset_name, num_classes=24):
    # TODO
    # This should either be moved to main.py or the arguments passed 
    # should be "args" instead. That way we avoid having to add unnecessary 
    # arguments as things come up.
    data_reader = {
            'DeepSig_Synthetic' : DeepSig_Synthetic,
            'DeepSig_2016a' : DeepSig_2016a,
            'DeepSig_2018'  : partial(DeepSig_2018, num_classes=num_classes),
            'DeepSig_2018_100' : DeepSig_2018_100,
            'DeepSig_2016cToa' : DeepSig_2016cToa
            }
    reader_fn = data_reader[dataset_name]

    return reader_fn


def download_DeepSig_data(fn):
    
    def _download_and_untar(url,out='./data'):
        os.makedirs(out, exist_ok=True)
        wget.download(
            url = url,
            out = out
        )
        _, file = os.path.split(url)
        dataset_fn = os.path.join(out,file)
        if file.endswith('bz2'):
            tar = tarfile.open(dataset_fn, "r:bz2")
        elif file.endswith('gz'):
            tar = tarfile.open(dataset_fn, "r:gz")
        tar.extractall(out)
        tar.close()
        print('dataset saved as {}'.format(dataset_fn))
        
    if not os.path.exists(fn):
        print('downloading data...')
        if '2016.10a' in fn:
            url = 'http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2'
            _download_and_untar(url)
        elif '2016.10b' in fn:
            url = url = 'http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2'
            _download_and_untar(url)
        elif '2018.01' in fn:
            url = 'http://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz'
            _download_and_untar(url)
        elif '2016.04C' in fn:
            url = 'http://opendata.deepsig.io/datasets/2016.04/2016.04C.multisnr.tar.bz2'
            _download_and_untar(url)

def normalize_unit_energy(train,X_test):
    def norm_list(L):
        def norm_sample(x):
            sig,label = x
            u = (sig/np.linalg.norm(sig),label)
            return u
        out = list(map(lambda x: norm_sample(x), L))
        return out
    train = norm_list(train)
    for k,v in X_test.items():
        X_test[k] = norm_list(v)
    return train, X_test

def amp_phase(train,X_test):
    
    def ap(L):
        def ap_samp(x):
            sig,label = x
            
            transpose = False
            if sig.shape[1] == 2:
                transpose = True
                sig = sig.T
            amp = np.linalg.norm(sig,axis=0)
            # ARG.... not needed since arctan2 takes care of division by 0. 
            # # make sure you don't divide by too small of a number.
            # eps = 1e-8
            # inds = np.abs(sig[0]) < eps
            # min_s0 = np.sign(sig[0][inds])*eps
            # sig[0][inds] = min_s0
            # q = sig[1]/sig[0]
            phase = np.arctan2(sig[1],sig[0])
            out = np.vstack([amp,phase])
            if transpose:
                out = out.T
                
            return (out,label)
        L = list(map(lambda x: ap_samp(x), L))
        return L 
        
    train = ap(train)
    for k,v in X_test.items():
        #print(k)
        X_test[k] = ap(v)    
    return train, X_test
            
        
            
def _normalize(train,X_test):
    X_train,y_train = zip(*train)
    X_train = np.array(X_train)
    def _get_mean_and_std(channel):
        mu = np.mean(channel,axis=0)
        sig = np.std(channel,axis=0)
        return mu,sig
    MUs = []; Sigs = []
    for i in range(2):
        channel = X_train[:,i,:].reshape(-1)
        mu, sig = _get_mean_and_std(channel)
        MUs.append(mu); Sigs.append(sig)
        X_train[:,i,:] = np.array(list(map(lambda x: (x-mu)/sig,
               X_train[:,i,:])))
         
    for k,v in X_test.items():
        X,y = list(zip(*v))
        X = np.array(X)
        for i in range(2):
            X[:,i,:] = np.array(list(map(lambda x: (x-MUs[i])/Sigs[i], 
             X[:,i,:])))
        X_test[k] = list(zip(X,y))
    return list(zip(X_train,y_train)), X_test
            
def DeepSig_2016a(data_path, train_split=0.5, seed=1234,
                  rescale=False, unit_energy=True, amp_phase=False):
    random.seed(seed)
    np.random.seed(seed)
    # reads in DeepSigs 2016 data file
    # Code mostly comes from here:
    # https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
    
    # download the data if necessary
    if data_path is None or not os.path.exists(data_path):
        print('downloading the data into ./data')
        download_DeepSig_data("./data/RML2016.10a_dict.pkl")
        data_path = './data/RML2016.10a_dict.pkl'
        
    # read in the data
    with open(data_path,'rb') as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'latin1'
        Xd = u.load()
        # puts all of the data into a list, X, s.t. X[modulation][snr]
        # e.g. X[k] is 2x128 and lbl[k] = ('CPFSK','4')
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], \
                                                      Xd.keys())))), [1,0])
        
        X_train = []
        y_train = []
        X_test = dict()
        for mod in mods:
            for snr in snrs:
                mod_snr_data = Xd[(mod,snr)]
                num_samples = mod_snr_data.shape[0]
                perm = np.random.permutation(num_samples)
                idx = int(np.round(train_split*num_samples))
                train_inds = perm[:idx]
                test_inds = perm[idx:]
                X_train.append([mod_snr_data[i] for i in train_inds])
                y_train.append([mods.index(mod) for i in train_inds])
                X_test[(mod,snr)] = [(mod_snr_data[i],mods.index(mod)) for \
                       i in test_inds]
                
        X_train = [item for sublist in X_train for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        mod_labels = dict(zip(mods,range(len(mods))))
 
        train = list(zip(X_train,y_train))
        if rescale:
            train, X_test = _normalize(train,X_test)
        if unit_energy:
            train, X_test = normalize_unit_energy(train,X_test)
        if amp_phase:
            train, X_test = amp_phase(train,X_test)
                        
   
        # shuffle just in case the data loader doesn't 
        out = dict()
        out['train_data'] = [train[i] for i in np.random.permutation(len(train))]
        out['test_data'] = X_test
        out['mods'] = mods
        out['snrs'] = snrs
        out['mods'] = mod_labels
        
        return out


def DeepSig_2016cToa(data_path, train_split=0.5, seed=1234,
                     rescale=False, unit_energy=True, amp_phase=False):
    # download the data if necessary
    train_data = os.path.join(data_path,'2016.04C.multisnr.pkl')
    test_data  = os.path.join(data_path,'RML2016.10a_dict.pkl')

    if train_data is None or not os.path.exists(train_data):
        print('downloading the data into ./data')
        download_DeepSig_data("./data/2016.04C.multisnr.pkl")
        train_data = './data/2016.04C.multisnr.pkl'

    if test_data is None or not os.path.exists(test_data):
        print('downloading the data into ./data')
        download_DeepSig_data("./data/RML2016.10a_dict.pkl")
        test_data = './data/RML2016.10a_dict.pkl'
        
    # read in the training data
    with open(train_data,'rb') as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'latin1'
        Xd = u.load()
        # puts all of the data into a list, X, s.t. X[modulation][snr]
        # e.g. X[k] is 2x128 and lbl[k] = ('CPFSK','4')
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], \
                                                      Xd.keys())))), [1,0])
        
        X_train = []
        y_train = []
        for mod in mods:
            for snr in snrs:
                mod_snr_data = Xd[(mod,snr)]
                X_train.append(mod_snr_data)
                y_train.append([mods.index(mod) for i in range(len(mod_snr_data))])

    
                
        X_train = [item for sublist in X_train for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        perm = np.random.permutation(len(X_train))
        X_train = [X_train[i] for i in perm]
        y_train = [y_train[i] for i in perm]
        mod_labels = dict(zip(mods,range(len(mods))))
        train = list(zip(X_train,y_train))

    with open(test_data,'rb') as fid:
        u = pickle._Unpickler(fid)
        u.encoding = 'latin1'
        X_test = u.load()
        # puts all of the data into a list, X, s.t. X[modulation][snr]
        # e.g. X[k] is 2x128 and lbl[k] = ('CPFSK','4')
        test_snrs,test_mods = map(lambda j: sorted(list(set(map(lambda x: x[j], \
                                                    Xd.keys())))), [1,0])
        assert test_snrs == snrs
        assert test_mods == mods 
        for mod in mods:
            for snr in snrs:
                x = X_test[(mod,snr)]
                x_in = [(x[i],mods.index(mod)) for i in range(len(x))]
                X_test[(mod,snr)] = x_in

    if rescale:
        train, X_test = _normalize(train,X_test)
    if unit_energy:
        train, X_test = normalize_unit_energy(train,X_test)
    if amp_phase:
        train, X_test = amp_phase(train,X_test)
    # shuffle just in case the data loader doesn't 
    out = dict()
    out['train_data'] = train
    out['test_data'] = X_test
    out['mods'] = mods
    out['snrs'] = snrs
    out['mods'] = mod_labels
    return out

def get_basic_classes(X, Y, Z, num_classes):

    all_classes = ["32PSK", "16APSK", "32QAM", "FM", "GMSK", "32APSK", "OQPSK", "8ASK", "BPSK", "8PSK", "AM-SSB-SC", 
                "4ASK", "16PSK", "64APSK", "128QAM", "128APSK", "AM-DSB-SC", "AM-SSB-WC", "64QAM", "QPSK", "256QAM", 
                "AM-DSB-WC", "OOK", "16QAM"]
    normal_class_mask = np.zeros(24, dtype=np.int32)

    if num_classes == 11:
        normal_classes = ["OOK", "4ASK", "BPSK", "QPSK", "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

        for class_ in normal_classes:
            normal_class_mask[all_classes.index(class_)] = 1

        classes = normal_classes
    else:
        normal_class_mask[:num_classes] = 1
        classes = all_classes[:num_classes]

    def is_from_normal_dataset(index):
        if np.sum(np.multiply(Y[index].astype(np.int32), normal_class_mask)) >= 1:
            return index

        return -1
    
    indices = range(len(Y))
    basic_indices = [is_from_normal_dataset(idx) for idx in tqdm(indices)]
    basic_indices = [i for i in basic_indices if i != -1]

    class_map = dict(zip(np.where(normal_class_mask == 1)[0].tolist(), range(num_classes)))

    X_basic = np.zeros((len(basic_indices), 1024, 2), dtype=np.float32)
    Y_basic = np.zeros((len(basic_indices), 11), dtype=np.int32)
    Z_basic = np.zeros((len(basic_indices), 1), dtype=np.int32)

    def populate_dset(args):
        i, index = args

        X_basic[i] = X[index]
        Y_basic[i][class_map[np.where(Y[index] == 1)[0][0]]] = 1
        Z_basic[i] = Z[index]

    for i, idx in tqdm(enumerate(basic_indices)):
        populate_dset((i, idx))

    return X_basic, Y_basic, Z_basic, classes
    
        
def DeepSig_2018(data_path="./data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5",
                 train_split=0.5,
                 seed=1234, rescale=False, unit_energy=True, amp_phase=False,
                 num_classes=24):
    random.seed(seed)
    np.random.seed(seed)

    # download the data if necessary
    if data_path is None or not os.path.exists(data_path):
        print('downloading the data into ../data')
        default_path = './data/2018.01' 
        download_DeepSig_data(default_path)
        data_path = os.path.join('./data/2018.01', 'GOLD_XYZ_OSC.0001_1024.hdf5')

    # read in the data
    dataset_path = data_path
    classes_path = os.path.join(os.path.dirname(data_path), 'classes.txt')

    assert os.path.isfile(dataset_path) 
    dataset = h5py.File(dataset_path, 'r')
    X, Y, Z = dataset.get('X'), dataset.get('Y'), dataset.get('Z')
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    dataset.close()

    classes = open(classes_path, "r").read().split('[')[1]
    classes = ''.join([c for c in classes if c not in "\'\"], "]).splitlines()

    if num_classes < 24:
        X, Y, Z, classes = get_basic_classes(X, Y, Z, num_classes)

    mods = classes
    snrs = np.unique(Z)
    t0 = time()
    train_data = []
    test_data = dict()
    for mod in mods:
        y = np.zeros(Y.shape[1],dtype='int')
        y[mods.index(mod)] = 1
        mod_inds = np.where((Y==y).all(axis=1))[0]
        for snr in snrs:
            snr_inds = np.where((Z==snr).all(axis=1))[0]
            inds = np.array(list(set(snr_inds).intersection(set(mod_inds))))
            perm = np.random.permutation(inds)
            idx = int(train_split*len(perm))
            train_inds = perm[:idx]
            test_inds  = perm[idx:]
            train_data.append(list(zip(X[train_inds],Y[train_inds])))
            test_data[(mod,snr)] = list(zip(X[test_inds],Y[test_inds]))
    
    train_data = [i for sublist in train_data for i in sublist]    
    random.shuffle(train_data)
    print('rt: {:0.2f}'.format(time()-t0))
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
        
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
    if unit_energy:
        train_data, test_data = normalize_unit_energy(train_data,test_data)
    if amp_phase:
        train_data, test_data = amp_phase(train_data,test_data)
    # TODO: Need to do re-scaling
    out = dict()
    out['train_data']  = train_data
    out['test_data'] = test_data
    out['mods'] = mods
    out['snrs'] = snrs
    return out


def build_small_DeepSig_2018(data_path="./data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5",
                             train_split=.5, rescale=False, unit_energy=True, 
                             amp_phase=False, num_samples=100,
                             outfile='DeepSig_2018_100.pkl'):
    # read in the data
    dataset_path = data_path
    classes_path = os.path.join(os.path.dirname(data_path), 'classes.txt')

    assert os.path.isfile(dataset_path) 
    dataset = h5py.File(dataset_path, 'r')
    X, Y, Z = dataset.get('X'), dataset.get('Y'), dataset.get('Z')
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    dataset.close()

    classes = open(classes_path, "r").read().split('[')[1]
    classes = ''.join([c for c in classes if c not in "\'\"], "]).splitlines()

    mods = classes
    snrs = np.unique(Z)
    t0 = time()
    train_data = []
    test_data = dict()
    for mod in mods:
        y = np.zeros(Y.shape[1],dtype='int')
        y[mods.index(mod)] = 1
        mod_inds = np.where((Y==y).all(axis=1))[0]
        for snr in snrs:
            snr_inds = np.where((Z==snr).all(axis=1))[0]
            inds = np.array(list(set(snr_inds).intersection(set(mod_inds))))
            perm = np.random.permutation(inds)[:num_samples]
            idx = int(train_split*len(perm))
            train_inds = perm[:idx]
            test_inds  = perm[idx:]
            train_data.append(list(zip(X[train_inds],Y[train_inds])))
            test_data[(mod,snr)] = list(zip(X[test_inds],Y[test_inds]))
    
    train_data = [i for sublist in train_data for i in sublist]    
    random.shuffle(train_data)
    print('rt: {:0.2f}'.format(time()-t0))
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
    if unit_energy:
        train_data, test_data = normalize_unit_energy(train_data,test_data)
    if amp_phase:
        train_data, test_data = amp_phase(train_data,test_data)
    # TODO: Need to do re-scaling
    out = dict()
    out['train_data']  = train_data
    out['test_data'] = test_data
    out['mods'] = mods
    out['snrs'] = snrs
    with open(os.path.join(os.path.dirname(data_path),outfile),'wb') as fid:
        pickle.dump(out,fid)
    return out    


def DeepSig_2018_100(data_path="./data/2018.01/DeepSig_2018_100.pkl",
                     train_split=0.5,
                     seed=1234, rescale=False, unit_energy=True, 
                     amp_phase=False,
                     num_classes=24):
    
    with open(data_path,'rb') as fid:
        out = pickle.load(fid)

    train_data = out['train_data']
    test_data = out['test_data']
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
        
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
    if unit_energy:
        train_data, test_data = normalize_unit_energy(train_data,test_data)
    if amp_phase:
        train_data, test_data = amp_phase(train_data,test_data)

    out['train_data']  = train_data
    out['test_data'] = test_data
      
    return out
    

def rml_2016b_reader():
    raise NotImplementedError


def rml_2016c_reader():
    raise NotImplementedError


def DeepSig_Synthetic(data_path='./data',samples_per_snr_class=100, 
                      train_split=0.5,n_classes=10,seed=1234, 
                      rescale=False, unit_energy=True, amp_phase=False):
    random.seed(seed)
    np.random.seed(seed)
    
    def _add_noise(signal,snr_db):
        # https://www.dsprelated.com/showcode/263.php
        # this is doing (a+ib)*(a-ib)
        N = signal.shape[1]
        signal_power = np.sum(signal**2)/N
        noise = np.random.randn(*signal.shape)
        noise_power = np.sum(noise**2)/N
        scale = (signal_power/noise_power)*(10**-(snr_db/10))
        adjusted_noise = np.sqrt(scale)*noise
        noisy_signal = signal + adjusted_noise
        return noisy_signal
    
    def _make_data(snr):
    
        # X, y = make_classification(n_samples=samples_per_snr_class, 
        #                             n_features=1024, n_classes=10, 
        #                             n_informative=10)
        X = np.random.randn(n_classes,1024)
        X = np.tile(X,(samples_per_snr_class//n_classes,1))
        y = np.tile(np.arange(0,n_classes),(samples_per_snr_class//n_classes,1)).astype('int64')
        y = y.reshape(-1)
        X = np.array(list(map(lambda x: np.tile(x,(1,2)).reshape(2,-1), X)))
        
        X = np.array(list(map(lambda x: _add_noise(x,snr), X)))
        X = X.astype('float32')
        ind = int(train_split*len(X))
        train_data = list(zip(*(X[:ind],y[:ind])))
        test_data  = list(zip(*(X[ind:],y[ind:])))
        return train_data, test_data  
    
    np.random.seed(seed=seed)
    snrs = np.arange(-20,20,2)
    mods = ['synthetic_C{:d}'.format(d) for d in range(10)]
    train_data = []
    test_data  = dict()
    for snr in snrs:
        for mod in mods:
            tr, te = _make_data(snr)
            train_data.append(tr)
            test_data[(mod,snr)] = te
            
    train_data = [i for sublist in train_data for i in sublist]       
    if rescale:
        train_data, test_data = _normalize(train_data,test_data)
    if unit_energy:
        train_data, test_data = normalize_unit_energy(train_data,test_data)
    if amp_phase:
        train_data, test_data = amp_phase(train_data,test_data)
    out = dict()
    out['train_data']  = train_data
    out['test_data'] = test_data
    out['mods'] = mods
    out['snrs'] = snrs
    return out








# def deepsig_2018_reader(dset_path, train_perc=0.5, seed=2018):
    # random.seed(seed)
    # np.random.seed(seed)

    # dataset = h5py.File(dset_path)
    # X, Y, Z = dataset.get('X'), dataset.get('Y'), dataset.get('Z') 
    # classes = open(os.path.join(os.path.dirname(dset_path), "classes.txt"), "r").read().splitlines()

    # X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    # dataset.close()

    # # This line takes a long time. My guess is that it's allocating 
    # # everything in memory. 
    # X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    # dataset.close()

    # # Get all the unique S/N ratios
    # unique_sn = np.unique(Z)

    # split_indices = []

    # # Used to center data & scale to unit variance
    # scaler = StandardScaler()

    # for sn in unique_sn:
        # indices = np.where(Z == sn)[0].tolist()
        # random.shuffle(indices)
        
        # train_indices = indices[:int(len(indices) * train_perc)]
        # test_indices = indices[int(len(indices) * train_perc):]
        
        # split_indices += [(train_indices, test_indices)]
        
        # scaler.partial_fit(np.reshape(X[train_indices], (-1, 1024 * 2)))

    # # For each Signal-to-Noise ratio, split the dataset into train / split. 
    # # We then concatenate each of these datasets to form our final dataset.
    # train_data, test_data = [], []

    # for train_indices, test_indices in split_indices:
        
        # X_train_scaled = scaler.transform(np.reshape(X[train_indices], (-1, 1024*2)))
        # X_train_scaled = np.reshape(X_train_scaled, (-1, 1024, 2))
        
        # train_data += [torch.utils.data.TensorDataset(torch.tensor(X_train_scaled), 
                                                    # torch.tensor(Y[train_indices]), 
                                                    # torch.tensor(Z[train_indices]))]
        
        # X_test_scaled = scaler.transform(np.reshape(X[test_indices], (-1, 1024*2)))
        # X_test_scaled = np.reshape(X_test_scaled, (-1, 1024, 2))
        
        # test_data += [torch.utils.data.TensorDataset(torch.tensor(X_test_scaled), 
                                                    # torch.tensor(Y[test_indices]), 
                                                    # torch.tensor(Z[test_indices]))]
        
    # # Concatenate all the various SN datasets into one large training set
    # train_dataset = torch.utils.data.ConcatDataset(train_data)
    # test_dataset  = torch.utils.data.ConcatDataset(test_data)

    # return train_dataset, test_dataset
