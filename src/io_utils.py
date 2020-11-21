import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import math

from src import s3_utils


def plot_sample_files_from_list(file_list, folder='', plot_sample=12, cols=6):
    """
    Plots random sample of images read from file_list
    :param file_list: list of files to print images from
    :param folder: path to files, defaults to current directory
    :param plot_sample: number of images to display
    :param cols: number of columns to arrange images
    :return: None
    """
    if plot_sample is None:
        plot_sample = len(file_list)
        
    imgs_to_plot = read_imgs_from_list(
        np.random.choice(file_list, min(plot_sample, len(file_list)), replace=False), 
        folder=folder
    )
    
    plot_sample_imgs(imgs_to_plot, plot_sample=plot_sample, cols=cols)


def plot_sample_files_from_s3(key, plot_sample=12, cols=6):
    """
    Plots random sample of images from s3 key provided
    :param key: path in s3
    :param plot_sample: number of imgs to display
    :param cols: number of columns to arrange images
    :return: None
    """
    file_list = s3_utils.get_image_list_from_s3(key)

    if plot_sample is None:
        plot_sample = len(file_list)

    files_to_plot = np.random.choice(file_list, min(plot_sample, len(file_list)), replace=False)
    imgs_to_plot = [s3_utils.read_image_from_s3(file) for file in files_to_plot]

    plot_sample_imgs(imgs_to_plot, plot_sample=plot_sample, cols=cols)


def read_imgs_from_list(file_list, folder=''):
    """
    Reads images from file_list
    :param file_list: list of files to read images
    :param folder: folder of files; defaults to current
    :return: list of images
    """
    if not folder:
        return [
            cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            for img_file in
            file_list
        ]
    return [
        cv2.cvtColor(cv2.imread('{}/{}'.format(folder, img_file)), cv2.COLOR_BGR2RGB)
        for img_file in
        file_list
    ]
    

def plot_sample_imgs(img_list, cols=6, rows=None, plot_sample=None):
    """
    Given list of images plots a sample of them
    :param img_list: list of images
    :param cols: number of columns to arrange images
    :param rows: number of rows to arrange images
    :param plot_sample: number of images to plot
    :return: None
    """
    if plot_sample is None:
        plot_sample = len(img_list)
       
    if (rows is None) or (rows == 0) or (rows > plot_sample):
        if (cols is None) or (cols == 0):
            rows = int(math.ceil(plot_sample ** 0.5))
            cols = int(math.ceil(plot_sample / rows))
        else:
            rows = int(math.ceil(plot_sample / cols))
    elif (cols is None) or (cols == 0) or (rows > plot_sample):
        cols = int(math.ceil(plot_sample / rows))

    if (rows == 0) or (cols == 0):
        print('Nothing to plot')
        return

    if rows * cols >= len(img_list):
        idx_plot = np.array(list(range(len(img_list))) + [None] * (rows * cols - len(img_list))).reshape(rows, cols)
    else:
        idx_plot = np.random.choice(range(len(img_list)), size=(rows, cols), replace=False)

    _, ax = plt.subplots(rows, cols, figsize=(16, (16 / cols) * rows))
    # 16 is the right width for my screen
    # height is calculated to keep same distance horizontally and vertically between plotted images

    # if rows ==1 or cols == 1 then plt.subplots(rows, cols) will create a vector of axis
    # otherwise subplots(roxws, cols) will create a matrix of axis
    
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
                if idx_plot[row][col] is not None:
                    ax[row, col].imshow(img_list[idx_plot[row][col]])
                else:
                    pylab.text(0.5, 0.5,'no image',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform = ax[row, col].transAxes
                        )
                __remove_ax_ticks(ax[row, col])
        
    plt.show()
    

def plot_metric(x, cut_off):
    """
    Given x, which is a list of numbers, sort them, plot them, with horizontal line at cut_off.
    This is to visualisize what gets selected and what not using cut_odd
    :param x: list of values
    :param cut_off: number to draw horizonal line at
    """
    plt.plot(sorted(x), label='Sorted metric')
    plt.axhline(cut_off, color='red', label='cut_off_point')
    plt.legend()
    plt.show()


def __remove_ax_ticks(ax):
    """
    Removes ticks from plot to plot images prettier
    :param ax: ax to remove ticks from
    """
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
