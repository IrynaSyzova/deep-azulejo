import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import math


def plot_sample_files(file_list, folder='', plot_sample=12, cols=6):
    if plot_sample is None:
        plot_sample = len(img_list)
        
    imgs_to_plot = read_imgs(
        np.random.choice(file_list, min(plot_sample, len(file_list)), replace=False), 
        folder=folder
    )
    
    plot_imgs(imgs_to_plot, plot_sample=plot_sample, cols=cols)


def read_imgs(file_list, folder=''):
    return [
        cv2.imread('{}/{}'.format(folder, img_file))[...,::-1]
        for img_file in
        file_list
    ]
    
    
def plot_imgs(img_list, cols=6, rows=None, plot_sample=None):
    
    if plot_sample is None:
        plot_sample = len(img_list)
    
    if (rows is None) or (rows == 0) or (rows > plot_sample):
        if (cols is None) or (cols == 0):
            rows = int(math.ceil(plot_sample ** 0.5))
            cols = int(math.ceil(plot_sample / rows))
        else:
            rows = int(math.ceil(plot_sample / cols))
    elif (cols is None) or (cols == 0) or (cols > plot_sample):
        cols = int(math.ceil(plot_sample / rows))
        
    if (rows == 0) or (cols == 0):
        print('Nothing to plot')
        return

    
    fig, ax = plt.subplots(rows, cols, figsize=(16, (16 / cols) * rows))
    # 16 is the right width for my screen
    # height is calculated to keep same distance horizontally and vertically between plotted images

    # if rows ==1 or cols == 1 then plt.subplots(rows, cols) will create a vector of axis
    # otherwise subplots(roxws, cols) will create a matrix of axis
    
    if rows * cols >= len(img_list):
        img_list_plot = np.array(img_list + [np.nan] * (rows * cols - len(img_list))).reshape(rows, cols)

    else:
        img_list_plot = np.random.choice(img_list, size=(rows, cols), replace=False)
    
    if rows == 1:
        for col in range(cols):
            ax[col].imshow(img_list[col])
            __remove_ax_ticks(ax[col])

    elif cols == 1:
        for row in range(rows):
            ax[row].imshow(img_list[row])
            __remove_ax_ticks(ax[row])
    else:
        for col in range(cols):
            for row in range(rows):
                if img_list_plot[row][col] is not np.nan:
                    ax[row, col].imshow(img_list_plot[row][col])
                else:
                    pylab.text(0.5, 0.5,'no image',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax[row, col].transAxes
                        )
                __remove_ax_ticks(ax[row, col])
        
    plt.show()
    

def plot_metric(x, cut_off):
    plt.plot(sorted(x), label='Sorted metric')
    plt.axhline(cut_off, color='red', label='cut_off_point')
    plt.legend()
    plt.show()


def __remove_ax_ticks(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)