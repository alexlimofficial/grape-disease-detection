"""
Utilities for model evaluation and test.


"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_metrics(model, generator, steps, target_names=[], name=None):
    """
    Function to print out confusion matrix and classification report.
    """
    target_names = ['black_measles', 'black_rot', 'leaf_blight', 'healthy']
    abbreviations = ['BM', 'BR', 'LB', 'H']
    
    # Get predictions for data
    y_pred = model.predict_generator(generator=generator, steps=steps)
    y_pred = np.argmax(a=y_pred, axis=1)
    
    # Get confusion matix
    cnf_mat = confusion_matrix(y_true=generator.classes, y_pred=y_pred)
    fig, ax = plt.subplots(1)
    ax = sns.heatmap(cnf_mat, ax=ax, cmap=plt.cm.Blues, annot=True, fmt='g')
    ax.set_xticklabels(abbreviations)
    ax.set_yticklabels(abbreviations)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()
    
    if name is not None:
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
        name = name + '.png'
        filename = os.path.join('./plots', name)
        plt.savefig(name)
        print('Saved confusion matrix as {}'.format(filename))
  
    # Get classification report
    print('Classification Report')
    print(classification_report(y_true=generator.classes, y_pred=y_pred, 
                              target_names=target_names))