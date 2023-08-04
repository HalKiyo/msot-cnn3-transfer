import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import colormaps as clm

from model import build_model

def main():
    #################################################
    #edit here
    #################################################
    overwrite_flag = False
    #################################################

    TRS = Transfer()
    TRS.evaluation(overwrite=overwrite_flag)
    plt.show()

class Transfer():
    def __init__(self):
        ###############################################################
        # change here
        ###############################################################
        self.old_epochs = 100
        self.old_batch_size = 256
        self.patience_num = 1000
        self.resolution = '1x1'
        ###############################################################
        # do not change here
        ###############################################################
        self.var_num = 1
        self.old_tors = 'predictors_coarse_std_Apr_o'
        self.old_tand = f"pr_{self.resolution}_std_MJJASO_thailand"
        self.new_tors = 'tos_coarse_std_Apr'
        self.new_tand = f"aphro_{self.resolution}_coarse_std_MJJASO_thailand"
        ###############################################################
        # do not change here
        ###############################################################
        self.lat, self.lon= 24, 72
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid * self.lon_grid
        self.loss = tf.keras.losses.MeanSquaredError()
        self.metrics = tf.keras.metrics.MeanSquaredError()
        ###############################################################
        # do not change here
        ###############################################################
        self.old_weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/ontinuous/" \
                               f"{self.old_tors}-{self.old_tand}"
        self.train_val_path = f"/docker/mnt/d/research/D3/cnn3/transfer/train_val/continuous/no_transfer/" \
                              f"{self.new_tors}-{self.new_tand}.pickle"
        self.result_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/result/continuous/thailand/" \
                          f"{self.resolution}/no_transfer"
        self.result_path = f"{self.result_dir}/" \
                           f"{self.new_tors}-{self.new_tand}.npy"

##############################################################################################################
############################## Evaluation Begin ################################################################

    def evaluation(self, overwrite=False):
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
                old_weights_path = f"{self.old_weights_dir}/" \
                               f"epoch{self.old_epochs}_batch{self.old_batch_size}_patience{self.patience_num}_{i}.h5"
                model.load_weights(old_weights_path)

                pred = model.predict(x_val) # (400, 1000)
                pred_lst.append(pred)

            pred_arr = np.array(pred_lst)
            np.save(self.result_path, pred_arr)
            print(f"{self.result_path}: prediction saved")
        else:
            matrix = []
            pred_arr = np.squeeze(np.load(self.result_path))
            for m in range(self.grid_num): # number of grid(true data)
                pred_arr_grid = pred_arr[m, :]
                corr = []
                for g in range(self.grid_num): # number of model(prediction)
                    y_val_grid = y_val[:, g]
                    corr_n = np.corrcoef(pred_arr_grid, y_val_grid)
                    corr.append(np.round(corr_n[0,1], 2))
                matrix.append(corr)
                print(f"model{m} has the best prediction skill in grid{np.nanargmax(corr)} at corr of {np.nanmax(corr)}")

                # draw r map
                sam = np.array(corr)
                sample = sam.reshape(20, 20)
                plt.imshow(sample, vmin=0, vmax=0.75, cmap=clm.temps)
                plt.colorbar()
                plt.show()

            matrix = np.array(matrix)

if __name__ == '__main__':
    main()
