import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def categorical_to_numpy(labels_in):
    """
    Converts a list of categorical labels to a numpy array with one-hot encoding.

    Parameters:
    labels_in (list): A list of categorical labels (e.g., ['dog', 'cat']).

    Returns:
    numpy.ndarray: A 2D array where each row represents one-hot encoding of the input labels.
    """
    labels = []
    for label in labels_in:
        if label == 'dog':
            labels.append(np.array([1, 0]))
        else:
            labels.append(np.array([0, 1]))
    return np.array(labels)

def one_hot_encoding(input):
    """
    Converts a numpy array of class indices to one-hot encoded format.

    Parameters:
    input (numpy.ndarray): A 1D array of class indices.

    Returns:
    numpy.ndarray: A 2D array of one-hot encoded values.
    """
    output = np.zeros((input.size, input.max() + 1))
    output[np.arange(input.size), input] = 1
    return output

def plot_one_image(data, labels, img_idx):
    """
    Plots a single image from a dataset.

    Parameters:
    data (numpy.ndarray): The dataset containing images.
    labels (list or numpy.ndarray): The labels for the images.
    img_idx (int): The index of the image to be plotted.

    Returns:
    None
    """
    my_img = data[img_idx, :].squeeze().reshape([32, 32, 3]).copy()
    my_label = labels[img_idx]
    print('label: %s' % my_label)
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(my_img.astype('uint8'), extent=[-1, 1, -1, 1])
    x_label_list = [0, 8, 16, 24, 32]
    y_label_list = [0, 8, 16, 24, 32]
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)
    fig.show(img)

def plot_acc(history, ax=None, xlabel='Epoch #'):
    """
    Plots the training and validation accuracy over epochs.

    Parameters:
    history (keras.callbacks.History): The history object returned by the fit method of keras models.
    ax (matplotlib.axes._subplots.AxesSubplot, optional): The matplotlib axis on which to plot. 
                                                          If None, a new axis is created.
    xlabel (str, optional): The label for the x-axis.

    Returns:
    None
    """
    history_dict = history.history
    history_dict.update({'epoch': list(range(len(history_dict['val_accuracy'])))})
    history_df = pd.DataFrame.from_dict(history_dict)

    best_epoch = history_df.sort_values(by='val_accuracy', ascending=False).iloc[0]['epoch']

    if not ax:
        f, ax = plt.subplots(1, 1)
    sns.lineplot(x='epoch', y='val_accuracy', data=history_df, label='Validation', ax=ax)
    sns.lineplot(x='epoch', y='accuracy', data=history_df, label='Training', ax=ax)
    ax.axhline(0.5, linestyle='--', color='red', label='Chance')
    ax.axvline(x=best_epoch, linestyle='--', color='green', label='Best Epoch')
    ax.legend(loc=7)
    ax.set_ylim([0.4, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    plt.show()
