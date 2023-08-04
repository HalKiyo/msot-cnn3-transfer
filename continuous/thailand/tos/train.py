import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import build_model
from util import transfer_load, _mask, train_val_split
from view import acc_map, show_map

def main():
    #################################################
    #edit here
    #################################################
    train_flag = True
    overwrite_flag = True
    new_patience_num = 2 # default=1000(no early stop)
    train_num = 16 # 16: 1958-1973
    val_num = 41 # 41: 1974-2014
    #################################################

    TRS = Transfer()
    if train_flag is True:
        predictors, predictand = transfer_load(TRS.new_tors, TRS.new_tand)
        train_inp, train_out, val_inp, val_out = train_val_split(predictors, 
                                                                 predictand, 
                                                                 train_num=train_num,
                                                                 val_num=val_num)
        TRS.training(train_inp, train_out, val_inp, val_out, patience_num=new_patience_num)
        print(f"{TRS.new_weights_dir}: SAVED")
        print(f"{TRS.train_val_path}: SAVED")
    else:
        print(f"train_flag is {train_flag}: not saved")

    TRS.validation(overwrite=overwrite_flag)
    TRS.show(val_index=TRS.val_index)
    plt.show()

class Transfer():
    def __init__(self):
        ###############################################################
        # change here
        ###############################################################
        self.frozen_num = 5
        self.val_index = 24 # 2011
        self.new_epochs = 30
        self.new_batch_size = 32
        self.resolution = '1x1'
        ###############################################################
        # do not change here
        ###############################################################
        self.var_num = 1
        self.new_tors = 'tos_coarse_std_Jul'
        self.new_tand = f"aphro_{self.resolution}_coarse_std_Aug_thailand"
        ###############################################################
        # do not change here
        ###############################################################
        self.seed = 1
        self.lat, self.lon= 24, 72
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid * self.lon_grid
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = tf.keras.metrics.MeanSquaredError()
        self.old_epochs = 100
        self.old_batch_size = 256
        self.old_patience_num = 1000
        self.old_tors = 'predictors_coarse_std_Apr_o'
        self.old_tand = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.old_weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/continuous/" \
                               f"{self.old_tors}-{self.old_tand}"
        ###############################################################
        # do not change here
        ###############################################################
        self.new_weights_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/weights/continuous/" \
                               f"{self.new_tors}-{self.new_tand}"
        self.train_val_path = f"/docker/mnt/d/research/D3/cnn3/transfer/train_val/continuous/" \
                              f"{self.new_tors}-{self.new_tand}.pickle"
        self.result_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/result/continuous/thailand/" \
                          f"{self.resolution}/{self.new_tors}-{self.new_tand}"
        self.result_path = f"{self.result_dir}/" \
                           f"/epoch{self.new_epochs}_batch{self.new_batch_size}.npy"

##############################################################################################################
############################## Training Begin ################################################################
    def training(self, x_train, y_train, x_val, y_val, patience_num=1000):
        # make input and output
        x_train = _mask(x_train)
        x_train = x_train[:, :, :, np.newaxis]
        x_val = _mask(x_val)
        x_val = x_val[:, :, :, np.newaxis]
        y_train = y_train.reshape(len(y_train), self.grid_num)
        y_train = _mask(y_train)
        y_val = y_val.reshape(len(y_val), self.grid_num)
        y_val = _mask(y_val)
        os.makedirs(self.new_weights_dir, exist_ok=True)

        # grid loop
        for i in range(self.grid_num):
            y_train_px = y_train[:, i]
            y_val_px = y_val[:, i]

            # model load
            model = build_model((self.lat, self.lon, self.var_num))
            old_weights_path = f"{self.old_weights_dir}/" \
                               f"epoch{self.old_epochs}_batch{self.old_batch_size}_patience{self.old_patience_num}_{i}.h5"
            model.load_weights(old_weights_path)

            # layer frozen 
            for layer_num in range(self.frozen_num):
                model.layers[layer_num].trainable = False

            # model setting
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])

            # early stop setting
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_num)
            his = model.fit(x_train, 
                            y_train_px, 
                            batch_size=self.new_batch_size, 
                            epochs=self.new_epochs,
                            validation_data=(x_val, y_val_px),
                            verbose=1,
                            callbacks=[early_stop]
                            )

            # save new_weights path
            new_weights_path = f"{self.new_weights_dir}" \
                               f"/epoch{self.new_epochs}_batch{self.new_batch_size}_{i}.h5"
            model.save_weights(new_weights_path)

        # save train_val pickle
        dct = {'x_train': x_train, 
               'y_train': y_train,
               'x_val': x_val, 
               'y_val': y_val,
               }
        with open(self.train_val_path, 'wb') as f:
            pickle.dump(dct, f)

############################## Training Done #################################################################
##############################################################################################################

    def validation(self, overwrite=False):
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        if os.path.exists(self.result_path) is False or overwrite is True:
            pred_lst = []
            corr = []
            os.makedirs(self.result_dir, exist_ok=True)
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                model = build_model((self.lat, self.lon, self.var_num))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                              loss=self.loss,
                              metrics=[self.metrics])
                new_weights_path = f"{self.new_weights_dir}/" \
                               f"epoch{self.new_epochs}_batch{self.new_batch_size}_{i}.h5"
                model.load_weights(new_weights_path)

                pred = model.predict(x_val) # (400, 1000)
                pred_lst.append(pred)

                pred = model.predict(x_val)
                corr_i = np.corrcoef(pred[:,0], y_val_px)
                corr.append(np.round(corr_i[0,1], 2))
                print(f"Correlation Coefficient of pixel{i}: {np.round(corr_i[0,1], 2)}")

            pred_arr = np.array(pred_lst)
            np.save(self.result_path, pred_arr)
        else:
            corr = []
            pred_arr = np.squeeze(np.load(self.result_path))
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                corr_i = np.corrcoef(pred_arr[i, :], y_val_px)
                corr.append(np.round(corr_i[0,1], 2))

        corr = np.array(corr)
        corr = corr.reshape(self.lat_grid, self.lon_grid)
        acc_map(corr)

    def show(self, val_index=0):
        # show true rain
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[val_index].reshape(self.lat_grid, self.lon_grid)
        y_val_px_masked = _mask(y_val_px)
        show_map(y_val_px_masked)

        # show predicted rain
        pred_lst = []
        if os.path.exists(self.result_path) is True:
            pred_val = np.squeeze(np.load(self.result_path))
            pred_arr = pred_val[:, val_index]
        else:
            for i in range(self.grid_num):
                model = build_model((self.lat, self.lon, self.var_num))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                              loss=self.loss,
                              metrics=[self.metrics])
                new_weights_path = f"{self.new_weights_dir}" \
                                   f"/epoch{self.new_epochs}_batch{self.new_batch_size}_{i}.h5"
                model.load_weights(new_weights_path)

                pred = model.predict(x_val)
                result = pred[val_index]
                pred_lst.append(result)
            pred_arr = np.array(pred_lst)
        pred_arr = pred_arr.reshape(self.lat_grid, self.lon_grid)
        pred_arr_masked = _mask(pred_arr)
        show_map(pred_arr_masked)

if __name__ == '__main__':
    main()
