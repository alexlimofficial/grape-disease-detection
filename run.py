"""
Top-level code to train models or apply visualizations/evaluation.

Author: Alex Lim

How to use:

Training:
python3 run.py -d </path/to/dataset> -m <modelname>

Testing:
python3 run.py -d </path/to/dataset> -m <modelname> -s <saved_model>
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import config
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.dataset import Dataset
from utils import evalutils, plotutils
from models import vanilla, vgg16, mobilenetv2, inceptionv3, nasnet


def run(datapath, model, saved_model_path=None):
    # Construct dataset pipeline
    dataset = Dataset(datapath).data
    
    # Base name for files
    basename = model.name + '_adam_lr{}_relu_dropout_epochs30'.format(str(config.learning_rate))

    if saved_model_path is not None:
        test_gen = dataset['test']
        num_instances = dataset['num_test']
    else:
        # Train model on dataset
        model.summary()
        model.train(dataset)
        
        # Save model
        modelname = basename + '.h5'
        saved_model_path = model.save(modelname)

        # Save plots
        plotname = basename + '_plt' # + additional tags
        plotutils.get_plots(model, smooth=True, name=plotname)

        test_gen = dataset['valtest']
        num_instances = dataset['num_val']

    # Save confusion matrix
    mtxname = basename + '_mtx'
    model.evaluate(
        saved_model_path,
        test_gen,
        num_instances,
        name=mtxname
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datapath', required=True)
    ap.add_argument('-m', '--model', required=True)
    ap.add_argument('-s', '--saved_model', required=False, default=None)
    args = vars(ap.parse_args())

    if args['model'] == 'vanilla':
        model = vanilla.VanillaCNN()
    elif args['model'] == 'vgg16':
        model = vgg16.VGG16()
    elif args['model'] == 'mobilenetv2':
        model = mobilenetv2.MobileNetV2()
    elif args['model'] == 'inceptionv3':
        model = inceptionv3.InceptionV3()
    elif args['model'] == 'nasnet':
        model = nasnet.NASNet()
    else:
        raise ValueError

    run(args['datapath'], model, args['saved_model'])