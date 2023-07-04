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

def shuffle(indata, outdata, vsample, seed=1, lat_grid=4, lon_grid=4):
    rng = np.random.default_rng(seed)

    outdata = outdata.reshape(42, 165, lat_grid, lon_grid)
    random_number = indata.shape[1]*indata.shape[2]
    random_index = rng.choice(random_number, random_number, replace=False)

    train_index = random_index[:-vsample]
    train_dct = {'model': train_index//indata.shape[2],
                 'year': train_index%indata.shape[2]}
    x_train = np.array([ indata[:, m, y] for m, y in zip(
                         train_dct['model'], train_dct['year']) ])
    y_train = np.array([ outdata[m, y, :, :] for m, y in zip(
                         train_dct['model'], train_dct['year']) ])

    val_index = random_index[-vsample:]
    val_dct = {'model': val_index//indata.shape[2],
                 'year': val_index%indata.shape[2]}
    x_val = np.array([ indata[:, m, y] for m, y in zip(
                         val_dct['model'], train_dct['year']) ])
    y_val = np.array([ outdata[m, y, :, :] for m, y in zip(
                         val_dct['model'], train_dct['year']) ])
    return x_train, y_train, x_val, y_val, train_dct, val_dct

