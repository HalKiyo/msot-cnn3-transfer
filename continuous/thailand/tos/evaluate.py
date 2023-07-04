import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

from util import open_pickle
from model import init_model
from view import ae_bar, TF_bar, draw_roc_curve, acc_map, bimodal_dist
import matplotlib.pyplot as plt

def main():
    EVAL = evaluate()
    """
    pred: (400, 1000)
    y_val: (1000, 400)
    """
    x_val, y_val, pred = EVAL.load_pred()
    if EVAL.mae_view_flag is True:
        EVAL.mae_evaluation(pred, y_val)
    if EVAL.rmse_view_flag is True:
        EVAL.rmse_evaluation(pred, y_val)
    if EVAL.true_false_view_flag is True:
        EVAL.true_false_bar(pred, y_val)
    if EVAL.auc_view_flag is True:
        roc = EVAL.auc(pred.T, y_val)
        draw_roc_curve(roc)
    if EVAL.corr_view_flag is True:
        EVAL.correlation(pred, y_val)
    plt.show()

class evaluate():
    def __init__(self):
        self.val_index = 20
        self.epochs =100
        self.batch_size =256
        self.seed =1
        self.vsample = 1000
        self.resolution = '1x1'
        ###############################################################
        # if you wanna change variables, don't forget to adjust var_num
        ###############################################################
        self.var_num = 4
        self.tors = 'predictors_coarse_std_Apr_msot'
        self.tant = f"pr_{self.resolution}_std_MJJASO_thailand"
        ###############################################################
        # path
        self.workdir = '/docker/mnt/d/research/D2/cnn3'
        self.train_val_path = self.workdir + f"/train_val/continuous/{self.tors}-{self.tant}.pickle"
        self.weights_dir = self.workdir + f"/weights/continuous/{self.tors}-{self.tant}"
        self.result_dir = self.workdir + f"/result/continuous/thailand/{self.resolution}/{self.tors}-{self.tant}"
        self.result_path = self.result_dir + f"/epoch{self.epochs}_batch{self.batch_size}_seed{self.seed}.npy"

        # model
        self.lat, self.lon = 24, 72
        self.lr = 0.001
        self.lat_grid, self.lon_grid = 20, 20
        self.grid_num = self.lat_grid*self.lon_grid
        # init_model is allowd to be called once otherwise layer_name will be messed up
        self.model = init_model(lat=self.lat, lon=self.lon, var_num=self.var_num, lr=self.lr)

        # view
        self.overwrite_flag = False
        self.mae_view_flag = False
        self.rmse_view_flag = True
        self.true_false_view_flag = True
        self.auc_view_flag = True
        self.corr_view_flag = True

    def load_pred(self):
        x_val, y_val = open_pickle(self.train_val_path)
        if os.path.exists(self.result_path) is False or self.overwrite_flag is True:
            pred_lst = []
            for i in range(self.grid_num):
                weights_path = f"{self.weights_dir}/epoch{self.epochs}_batch{self.batch_size}_{i}.h5"
                model = self.model
                model.load_weights(weights_path)
                pred = model.predict(x_val)
                pred_lst.append(pred)
            pred_arr = np.squeeze(np.array(pred_lst))
            np.save(self.result_path, pred_arr)
        else:
            pred_arr = np.squeeze(np.load(self.result_path))
        return x_val, y_val, pred_arr # pred(400, 1000)

    def mae_evaluation(self, pred, y):
        value = pred[:, self.val_index] # pred(400, 1000)
        label = y[self.val_index, :] # y(1000, 400)
        ae = np.abs(value - label)
        ae_flat = ae.reshape(-1)
        mae = np.mean(ae_flat)
        print(mae)
        ae_bar(ae_flat)

    def rmse_evaluation(self, pred, y):
        rmse_flat = []
        for px in range(len(pred)):
            value = pred[px, :] # pred(400, 1000)
            label = y[:, px] # y(1000, 400)
            rmse = np.sqrt(np.mean((value - label)**2))
            rmse_flat.append(rmse)
        rmse_flat = np.array(rmse_flat)

        n = len(rmse_flat)
        sample_mean = np.mean(rmse_flat)
        sample_var = stats.tvar(rmse_flat)
        interval = stats.norm.interval(alpha=0.95,
                                       loc=sample_mean,
                                       scale=np.sqrt(sample_var/n))
        print(f"rmse_95%reliable_mean spans {interval}, {sample_mean}")

        rmse_map = rmse_flat.reshape(self.lat_grid, self.lon_grid)
        acc_map(rmse_map, vmin=0.10, vmax=0.35)

    def GMM(self, data):
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data.reshape(-1, 1)) # 次元数2を入力とするため変形
        estimated_group = gmm.predict(data.reshape(-1, 1))
        return gmm

    def true_false_bar(self, pred, y, criteria=0.1):
        true_count, false_count = 0, 0
        rmse_flat = []
        for sam in range(len(y)):
            value = pred[:, sam] # pred(400, 1000)
            label = y[sam, :] # y(1000, 400)
            rmse = np.sqrt(np.mean((value - label)**2))
            rmse_flat.append(rmse)

        rmse_flat = np.array(rmse_flat)
        gmm = self.GMM(rmse_flat)
        criteria = np.mean([gmm.means_[0, -1],
                            gmm.means_[1, -1]])

        for sam in range(len(y)):
            if rmse_flat[sam] <= criteria:
                true_count += 1
            else:
                false_count += 1

        print(f"mean of gmm is {criteria}")
        bimodal_dist(rmse_flat, gmm)
        TF_bar(true_count, false_count)

    def roc(self, sim, obs, percentile=20):
        """
        this roc function just returns single event
        if multiple events are neede to be evaluated,
        call auc function below
        """
        # percentile should be absolute number
        sim = np.abs(sim)
        obs = np.abs(obs)

        # make criteria
        sim_per = np.percentile(sim, percentile)
        obs_per = np.percentile(obs, percentile)

        # calculate number of obs percentile
        over_per = sum(obs > obs_per)
        under_per = sum(obs <= obs_per)

        # save count of hit and false pixcel
        hit_count = 0
        false_count = 0
        for p in range(len(obs)):
            if sim[p] > sim_per and obs[p] > obs_per:
                hit_count += 1
            elif sim[p] > sim_per and obs[p] <= obs_per:
                false_count += 1

        # calculate HitRate and FalseAlertRate
        hr = hit_count/over_per
        far = false_count/under_per

        return hr, far

    def auc(self, sim, obs):
        result = [[0,0]]
        # percentile variation list
        per_list = np.arange(10, 100, 10)
        per_list = per_list[::-1]

        # calsulate different percentile result
        for i in per_list:
            # calculate multiple varidation events
            hr_all, far_all = [], []
            for j in range(len(obs)):
                hr_n, far_n = self.roc(sim[j], obs[j], percentile=i)
                hr_all.append(hr_n)
                far_all.append(far_n)
            hr, far = np.mean(hr_all), np.mean(far_all)
            result.append([hr, far])

        result.append([1, 1])
        result = np.array(result)
        return result

    def correlation(self, pred, y_val):
        corr = []
        pred_arr = np.squeeze(pred)
        for i in range(self.grid_num):
            y_val_px = y_val[:, i]
            corr_i = np.corrcoef(pred_arr[i,:], y_val_px)
            corr.append(np.round(corr_i[0, 1], 2))
            #print(f"gird: {i}")

        # calculate 95% intervals
        n = len(corr)
        sample_mean = np.mean(corr)
        sample_var = stats.tvar(corr)
        interval = stats.norm.interval(alpha=0.95,
                                       loc=sample_mean,
                                       scale=np.sqrt(sample_var/n))
        print(f"corr_95%reliable_mean spans {interval}, {sample_mean}")

        # view corr heat-map
        corr = np.array(corr)
        corr = corr.reshape(self.lat_grid, self.lon_grid)
        acc_map(corr)


if __name__ == '__main__':
    main()
