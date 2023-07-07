import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import build_model
from util import transfer_load, _mask
from view import acc_map, show_map

def main():
    overwrite_flag = False

    TRS = Transfer()
    #TRS.save_pickle()
    #print(f"{TRS.train_val_path}: SAVED")
    #TRS.validation(overwrite=overwrite_flag)
    TRS.show(val_index=TRS.val_index)
    plt.show()

class Transfer():
    def __init__(self):
        ###############################################################
        # change here
        ###############################################################
        self.val_index = 57 # 0 = 1958
        self.resolution = '1x1'
        ###############################################################
        # do not change here
        ###############################################################
        self.var_num = 1
        self.new_tors = 'tos_coarse_std_Apr'
        self.new_tand = f"aphro_{self.resolution}_coarse_std_MJJASO_thailand"
        ###############################################################
        # do not change here
        ###############################################################
        self.seed = 1
        self.lat, self.lon= 24, 72
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid * self.lon_grid
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = tf.keras.metrics.MeanSquaredError()
        ###############################################################
        # no transfer weights
        ###############################################################
        self.old_epochs = 100
        self.old_batch_size = 256
        self.old_tors = 'predictors_coarse_std_Apr_o'
        self.old_tand = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.old_weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/continuous/" \
                               f"{self.old_tors}-{self.old_tand}"
        ###############################################################
        # transfer weights
        ###############################################################
        #self.old_epochs = 1
        #self.old_batch_size = 32
        #self.old_tors = 'tos_coarse_std_Apr'
        #self.old_tand = f"aphro_{self.resolution}_coarse_std_MJJASO_thailand"
        #self.old_weights_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/weights/continuous/" \
        #                       f"{self.old_tors}-{self.old_tand}"
        ###############################################################
        # do not change here
        ###############################################################
        self.train_val_path = f"/docker/mnt/d/research/D3/cnn3/transfer/train_val/continuous/no_transfer/" \
                              f"{self.new_tors}-{self.new_tand}.pickle"
        self.result_path = f"/docker/mnt/d/research/D3/cnn3/transfer/result/continuous/thailand/" \
                          f"{self.resolution}/no_transfer/{self.new_tors}-{self.new_tand}.npy"
        ###############################################################

    def save_pickle(self):
        predictors, predictand = transfer_load(self.new_tors, self.new_tand)
        x_val = predictors[:, :, :, np.newaxis]
        x_val = _mask(x_val)
        y_val = predictand.reshape(len(predictand), self.grid_num)
        y_val = _mask(y_val)
        dct = {'x_val': x_val,
               'y_val': y_val}
        with open(self.train_val_path, 'wb') as f:
            pickle.dump(dct, f)

    def validation(self, overwrite=False):
        with open(self.train_val_path, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']

        if os.path.exists(self.result_path) is False or overwrite is True:
            pred_lst = []
            corr = []
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                model = build_model((self.lat, self.lon, self.var_num))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                              loss=self.loss,
                              metrics=[self.metrics])
                old_weights_path = f"{self.old_weights_dir}/" \
                               f"epoch{self.old_epochs}_batch{self.old_batch_size}_{i}.h5"
                model.load_weights(old_weights_path)

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

        # show correlation map
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
                old_weights_path = f"{self.old_weights_dir}" \
                                   f"/epoch{self.old_epochs}_batch{self.old_batch_size}_{i}.h5"
                model.load_weights(old_weights_path)

                pred = model.predict(x_val)
                result = pred[val_index]
                pred_lst.append(result)
            pred_arr = np.array(pred_lst)
        pred_arr = pred_arr.reshape(self.lat_grid, self.lon_grid)
        pred_arr_masked = _mask(pred_arr)
        show_map(pred_arr_masked)

if __name__ == '__main__':
    main()
