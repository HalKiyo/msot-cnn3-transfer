import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

def draw_val(val_pred, val_label_onehot, class_num=5):
    plt.rcParams["font.size"] = 18
    fig = plt.figure()
    ax = plt.subplot()

    val_list = [i for i in range(class_num)]
    width, linewidth, align = 0.5, 0.5, 'center'

    # count
    val_true = []
    val_false = []
    for i, j in zip(val_pred, val_label_onehot):
        if np.argmax(i) == np.argmax(j):
            val_true.append(val_list[np.argmax(j)])
        else:
            val_false.append(val_list[np.argmax(j)])

    # true
    val_tcount = [val_true.count(j) for j in val_list]
    ax.bar(val_list, val_tcount,
           color='darkslategray', width=width, linewidth=linewidth, align=align)

    # false
    val_fcount = [val_false.count(j) for j in val_list]
    ax.bar(val_list, val_fcount,
           color='darkgoldenrod', bottom=val_tcount, width=width, linewidth=linewidth, align=align, alpha=.8)

    # count
    val_label_class =  [np.argmax(i) for i in val_label_onehot]
    u, counts = np.unique(val_label_class, return_counts=True)

    plt.show(block=False)
    return u, counts

def show_class(image, class_num=5, lat_grid=4, lon_grid=4, txt_flag=False):
    cmap = plt.cm.get_cmap('BrBG', class_num)
    bounds = [i -0.5 for i in range(class_num+1)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = [i for i in range(class_num)]

    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)
    dlt = 2
    txt_extent = (img_extent[0] + dlt, img_extent[1] - dlt,
                  img_extent[2] + dlt, img_extent[3] - dlt)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(image,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     norm=norm,
                     cmap=cmap)
    cbar = fig.colorbar(mat,
                        ax=ax,
                        extend='both',
                        ticks=ticks,
                        spacing='proportional',
                        orientation='vertical')
    if txt_flag is True:
        if class_num == 5:
            cbar.ax.set_yticklabels(['low', 'mid-low', 'normal', 'mid-high', 'high'])
        else:
            lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
            lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
            for i, lat in enumerate(lat_lst):
                for j, lon in enumerate(lon_lst):
                    ax.text(lon, lat, image[i,j],
                            ha="center", va="center", color='black', fontsize='15')
    plt.rcParams["font.size"] = 18

    plt.show(block=False)

def view_accuracy(acc, lat_grid=4, lon_grid=4):
    projection = ccrs.PlateCarree(central_longitude=180)
    img_extent = (-90, -70, 5, 25) # location = (N5-25, E90-110)
    dlt = 2
    txt_extent = (img_extent[0] + dlt, img_extent[1] - dlt,
                  img_extent[2] + dlt, img_extent[3] - dlt)

    fig = plt.figure()
    ax = plt.subplot(projection=projection)
    ax.coastlines()
    mat = ax.matshow(acc,
                     origin='upper',
                     extent=img_extent,
                     transform=projection,
                     vmin=0, vmax=1,
                     cmap='Oranges')

    lat_lst = np.linspace(txt_extent[3], txt_extent[2], lat_grid)
    lon_lst = np.linspace(txt_extent[0], txt_extent[1], lon_grid)
    for i, lat in enumerate(lat_lst):
        for j, lon in enumerate(lon_lst):
            ax.text(lon, lat, acc[i,j],
                    ha="center", va="center", color='white', fontsize='15')
    cbar = fig.colorbar(mat, ax=ax)
    plt.rcParams["font.size"] = 18
    plt.show(block=False)

def pred_accuracy(true_count, false_count):
    fig = plt.figure()
    ax = plt.subplot()
    print(true_count, false_count)
    ax.barh(1, true_count, height=0.5, color='darkslategray', align='center', label='True')
    ax.barh(1, false_count, left=true_count, height=0.5, color='orange', align='center', label='False')
    ax.set_ylim(0,2)
    ax.set_yticks([1.0], ['val'])
    plt.legend()
    plt.rcParams["font.size"] = 18
    plt.show(block=False)

def bimodal_dist(true_count_of_each_individual_validation_sample):
    fig, ax = plt.subplots()
    ax.hist(true_count_of_each_individual_validation_sample, color='g', alpha=0.5)
    plt.show(block=False)

def view_probability(pred, flag=False):
    fig = plt.figure()
    ax = plt.subplot()
    width, linewidth, align = 0.5, 0.5, 'center'
    if flag is False:
        color='darkgoldenrod'
    else:
        color='darkslategray'

    ticks = np.arange(len(pred))
    ax.bar(ticks,
           pred,
           color=color,
           width=width,
           linewidth=linewidth,
           align=align)
    ax.set_ylim(0,1)
    plt.rcParams["font.size"] = 18
    plt.show(block=False)

def box_crossentropy(true, false, class_num=5):
    plt.rcParams["font.size"] = 18
    label = np.arange(1)
    xs = {key:val for key, val in zip(true.keys(), label)}
    shift = 0.1

    fig, ax = plt.subplots()
    for key in true.keys():
        ax.boxplot(true[key], positions=[xs[key] - shift], boxprops=dict(color='darkslategray'),
                   showfliers=False)
        ax.boxplot(false[key], positions=[xs[key] + shift], boxprops=dict(color='darkgoldenrod'),
                   showfliers=False)
    ax.set_xticks(range(1))
    ax.set_xticklabels(true.keys())
    plt.show(block=False)
