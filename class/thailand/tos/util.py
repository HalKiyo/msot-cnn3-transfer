import pickle
import numpy as np
import numpy.ma as ma

def pretrained_load(inkey, outkey):
    infile = f"/docker/mnt/d/research/D2/cnn3/predictors/{inkey}.npy"
    outfile = f"/docker/mnt/d/research/D2/cnn3/predictant/class/{outkey}.npy"
    predictors = np.load(infile)
    predictant = np.load(outfile)
    return predictors, predictant

def transfer_load(inkey, outkey):
    """
    ORAS5.shape = (56, 24, 62), 1958-2014
    aphro.shape = (65, 20, 20), 1951-2015
    """
    infile = f"/docker/mnt/d/research/D3/cnn3/transfer/predictors/{inkey}.npy"
    outfile = f"/docker/mnt/d/research/D3/cnn3/transfer/predictand/class/{outkey}.npy"
    predictors = np.load(infile)
    predictand = np.load(outfile)
    tors_reshaped = predictors[:, :, :]
    tand_reshaped = predictand[7:64, :, :]
    return tors_reshaped, tand_reshaped

def _mask(x, fill_value=-99):
    m = ma.masked_where(x <= fill_value, x)
    z = ma.masked_where(m==0, m)
    f = ma.filled(z, 0)
    return f

def open_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    x_val, y_val = data['x_val'], data['y_val']
    return x_val, y_val

def train_val_split(indata, outdata, train_num=32, val_num=25):
    """
    train: 1958-1989 (32 years)
    validation: 1990-2014 (25 years)
    """
    train_inp = indata[:train_num, :, :]
    train_out = outdata[:train_num, :, :]
    val_inp = indata[-val_num:, :, :]
    val_out = outdata[-val_num:, :, :]
    return train_inp, train_out, val_inp, val_out

