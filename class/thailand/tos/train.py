import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model3 import build_model
from view import draw_val, show_class, view_accuracy
from util import transfer_load, train_val_split, _mask

def main():
    train_flag = False

    TRS = Transfer()
    predictors, predictand = transfer_load(TRS.new_tors, TRS.new_tand)
    print(predictors.shape, predictand.shape)
    exit()

    if train_flag is True:
        predictors, predictant = transfer_load(TRS.new_tors, TRS.new_tand)
        #px.training(*shuffle(predictors, predictant, px.vsample, px.seed, px.lat_grid, px.lon_grid))
        print(f"{TRS.new_weights_dir}: SAVED")
        print(f"{TRS.train_val_path}: SAVED")
    else:
        print(f"train_flag is {train_flag}: not saved")

    TRS.validation()
    TRS.show(val_index=TRS.val_index)
    TRS.label_dist_multigrid()

    plt.show()

class Transfer():
    def __init__(self):
        ###################################################
        # change here
        ###################################################
        self.val_index = 1 #true_index=330, false_index=20
        self.class_num = 5
        self.new_epochs = 200
        self.new_batch_size = 32
        self.descrete_mode = 'EFD'
        self.resolution = '1x1' # 1x1 or 5x5_coarse
        self.new_tors = 'tos_coarse_std_Apr'
        self.new_tand = f"aphro_{self.resolution}_coarse_std_MJJASO_thailand_{self.descrete_mode}_{self.class_num}"
        ###################################################
        # do not change here
        ###################################################
        self.seed = 1
        self.old_epochs = 150
        self.old_batch_size = 256
        self.old_tors = 'predictors_coarse_std_Apr_msot'
        self.old_tand = f"pr_{self.resolution}_std_MJJASO_thailand_{self.descrete_mode}_{self.class_num}"
        self.old_weights_dir = f"/docker/mnt/d/research/D2/cnn3/weights/class/" \
                               f"{self.old_tors}-{self.old_tand}"
        ###################################################
        # do not change here
        ###################################################
        self.var_num = 1
        self.lat, self.lon = 24, 72
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid 
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.metrics = tf.keras.metrics.CategoricalAccuracy()
        #####################################################
        # do not change here
        ####################################################
        self.train_val_path = f"/docker/mnt/d/research/D3/cnn3/transfer/train_val/class/" \
                              f"{self.new_tors}-{self.new_tand}.pickle"
        self.new_weights_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/weights/class/" \
                               f"{self.new_tors}-{self.new_tand}"
        self.result_dir = f"/docker/mnt/d/research/D3/cnn3/transfer/result/class/thailand/" \
                          f"{self.resolution}/{self.new_tors}-{self.new_tand}"
        self.result_path = f"{self.result_dir}" \
                           f"/class{self.class_num}_epoch{self.new_epochs}_batch{self.new_batch_size}.npy"

    ###########################################################
    # training begin
    ##########################################################
    def training(self, x_train, y_train, x_val, y_val, patience_num=1000):
        x_train = _mask(x_train)
        x_train = x_train[:, :, :, np.newaxis]
        x_val = _mask(x_val)
        x_val = x_val[:, :, :, np.newaxis]
        y_train = y_train.reshape(len(y_train), self.grid_num)
        y_val = y_val.reshape(len(y_val, self.grid_num))
        os.makedirs(self.new_weights_dir, exist_ok=True) # create weight directory

        for i in range(self.grid_num):
            y_train_px = y_train[:, i]
            y_train_one_hot = tf.keras.utils.to_categorical(y_train_px, self.class_num)
            model = build_model((self.lat, self.lon, self.var_num), self.class_num)
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                          loss=self.loss,
                          metrics=[self.metrics])
            his = model.fit(x_train, y_train_one_hot, batch_size=self.batch_size, epochs=self.epochs)
            weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.save_weights(weights_path)
        dct = {'x_train': x_train, 'y_train': y_train,
               'x_val': x_val, 'y_val': y_val,
               'train_dct': train_dct, 'val_dct': val_dct}
        with open(self.savefile, 'wb') as f:
            pickle.dump(dct, f)

    def validation(self):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        pred_lst = []
        acc = []
        for i in range(self.grid_num):
            y_val_px = y_val[:, i]
            y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
            model = build_model((self.lat, self.lon, self.var_num), self.class_num)
            model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                          loss=self.loss, 
                          metrics=[self.metrics])
            weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
            model.load_weights(weights_path)
            pred = model.predict(x_val) # (400, 1000, 5)
            pred_lst.append(pred)
            result = model.evaluate(x_val, y_val_one_hot)
            acc.append(round(result[1], 2))
            print(f"CategoricalAccuracy of pixcel{i}: {result[1]}")

        pred_arr = np.array(pred_lst)
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir, exist_ok=True) # create weight directory
        np.save(self.result_path, pred_arr)

        acc = np.array(acc)
        acc = acc.reshape(self.lat_grid, self.lon_grid)
        view_accuracy(acc)

    def show(self, val_index):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[val_index].reshape(self.lat_grid, self.lon_grid)
        show_class(y_val_px, class_num=self.class_num)

        pred_lst = []
        if os.path.exists(self.result_path) is True:
            pred_arr = np.load(self.result_path)
            for i in range(self.grid_num):
                label = np.argmax(pred_arr[i, val_index])
                pred_lst.append(label)
        else:
            for i in range(self.grid_num):
                model = build_model((self.lat, self.lon, self.var_num), self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.loss, 
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                label = np.argmax(pred[val_index])
                pred_lst.append(label)
                print(f"pixcel{i}: {label}")
        pred_label = np.array(pred_lst)
        pred_label = pred_label.reshape(self.lat_grid, self.lon_grid)
        show_class(pred_label, class_num=self.class_num)

    def label_dist(self, px_index):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_px = y_val[:, px_index]
        y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
        model = build_model((self.lat, self.lon, self.var_num), self.class_num)
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                      loss=self.loss, 
                      metrics=[self.metrics])
        weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{px_index}.h5"
        model.load_weights(weights_path)
        pred = model.predict(x_val)
        class_label, counts = draw_val(pred, y_val_one_hot, class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")

    def label_dist_multigrid(self):
        with open(self.savefile, 'rb') as f:
            data = pickle.load(f)
        x_val, y_val = data['x_val'], data['y_val']
        y_val_lst = []
        if os.path.exists(self.result_path) is True:
            pred_arr = np.load(self.result_path)
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                y_val_lst.append(y_val_one_hot)
        else:
            pred_lst = []
            for i in range(self.grid_num):
                y_val_px = y_val[:, i]
                y_val_one_hot = tf.keras.utils.to_categorical(y_val_px, self.class_num)
                model = build_model((self.lat, self.lon, self.var_num), self.class_num)
                model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                              loss=self.loss, 
                              metrics=[self.metrics])
                weights_path = f"{self.weights_dir}/class{self.class_num}_epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
                y_val_lst.append(y_val_one_hot)
            pred_arr = np.array(pred_lst)
        pred_arr = pred_arr.reshape(self.grid_num*self.vsample, self.class_num)
        y_val_arr = np.array(y_val_lst).reshape(self.grid_num*self.vsample, self.class_num)
        class_label, counts = draw_val(pred_arr, y_val_arr, class_num=self.class_num)
        print(f"class_label: {class_label}\n" \
              f"counts: {counts}")


if __name__ == '__main__':
    main()

