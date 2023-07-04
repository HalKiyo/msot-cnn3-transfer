import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import colormaps as clm
from matplotlib.colors import Normalize
from sklearn.metrics import auc
from scipy.stats import norm

def acc_map(acc, lat_grid=20, lon_grid=20, vmin=-0.10, vmax=0.60, discrete=7):
    plt.rcParams["font.size"] = 18
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)

    mpl.colormaps.unregister('sandbox')
    mpl.colormaps.register(clm.temps, name='sandbox')
    cm = plt.cm.get_cmap('sandbox', discrete)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=vmin, vmax=vmax,
                     cmap=cm)
    cbar = fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def show_map(image, vmin=-1, vmax=1):
    plt.rcParams["font.size"] = 18
    cmap = plt.cm.get_cmap('BrBG')

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # locatin = (N5-25, E90-110)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=Normalize(vmin=vmin, vmax=vmax),
                     cmap = cmap)
    cbar = fig.colorbar(mat, ax=ax)
    plt.show(block=False)

def ae_bar(data, vmin=0, vmax=2):
    # grid毎にabs(実際のlabelデータ-予測結果)を400個棒グラフにして出力する
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    ax = plt.subplot()
    pixcel = np.arange(len(data))
    ax.bar(pixcel, data, color='magenta')
    ax.set_ylim(vmin, vmax)
    plt.show(block=False)

def bimodal_dist(data, gmm):
    plt.rcParams["font.size"] = 18
    fig, ax = plt.subplots()

    # histgram
    ax.hist(data, color='lightsteelblue', alpha=0.8)

    # gaussian mixture modelling
    ax2 = ax.twinx()
    x = np.linspace(0, 1, 1000)
    true = norm.pdf(x,
                    gmm.means_[0, -1],
                    np.sqrt(gmm.covariances_[0]))
    false = norm.pdf(x,
                     gmm.means_[1, -1],
                     np.sqrt(gmm.covariances_[1]))
    ax2.plot(x,
             np.squeeze(gmm.weights_[0]*true),
             linestyle='dashed',
             color='darkslategray',
             label='true')
    ax2.plot(x,
             np.squeeze(gmm.weights_[1]*false),
             linestyle='solid',
             color='darkgoldenrod',
             label='false')
    ax2.legend()
    plt.show(block=False)

def TF_bar(true_count, false_count):
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    ax = plt.subplot()
    print(f"true: {true_count}, false: {false_count}")
    ax.barh(1,
            true_count, 
            height=0.5, 
            color='darkslategray', 
            align='center', 
            label=f"True({true_count})")
    ax.barh(1,
            false_count,
            left=true_count,
            height=0.5,
            color='darkgoldenrod',
            align='center',
            label=f"False({false_count})")
    ax.set_ylim(0,2)
    ax.set_yticks([1.0], ['validation(1000samples)'])
    plt.legend()
    plt.show(block=False)

def draw_roc_curve(roc):
    plt.rcParams["font.size"] = 18
    # calculate auc
    fpr = roc[:, 1]
    tpr = roc[:, 0]
    AUC = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))

    # draw cnn_continuous line
    plt.plot(fpr,
             tpr,
             label=f"ROC curve (AUC = {round(AUC, 3)})",
             color="deeppink",
             linestyle=":",
             linewidth=4)
    # plot cnn_contionuous percentile results
    plt.scatter(fpr, tpr, s=100, color='red')

    # plot auc=0.5 line
    plt.plot([0,1],
             [0,1],
             "k--",
             label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show(block=False)
