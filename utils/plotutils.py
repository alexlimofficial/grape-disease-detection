"""
Utilities for plotting.

Author: Alex Lim
"""
import os
import datetime
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.8):
    """
    Helper function to help smooth accuracy/loss plots using an exponential
    moving average of the loss and accuracy values. 
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def get_plots(model, smooth=False, name=None):
    """
    Helper function to plot training/validation accuracy/loss.
    @param history - Keras model history 
    @param smooth - Boolean for whether to smooth plots or not
    """
    history = model.history
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)
    
    if smooth == True:
        acc = smooth_curve(acc)
        val_acc = smooth_curve(val_acc)
        loss = smooth_curve(loss)
        val_loss = smooth_curve(val_loss)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc,'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.tight_layout()

    if name is not None:
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
        filename = name + '.png'
        plotpath = os.path.join('./plots', filename)
        plt.savefig(plotpath)
        print('Plot saved as {}'.format(plotpath))
