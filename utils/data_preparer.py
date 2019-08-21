"""
Dataset preparer class.

Author: Alex Lim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import random
import shutil
import argparse
from collections import OrderedDict

# Set random seed for consistency
random.seed(100)


class DatasetPreparer:
    """Class to prepare data and split into train/val/test
    in a format which is compatible with Keras. Expects data
    directory structure where folders represent unique class
    names and contains corresponding class images.
    """
    def __init__(self, datapath):
        self.datapath = datapath
        self.classes = None
        self.total_images = None
        self.num_instances = OrderedDict()
        self.num_train = OrderedDict()
        self.num_val = OrderedDict()
        self.num_test = OrderedDict()
        self.load_data()

    def load_data(self):
        """Gets all valid classes from data folder. Expects
        each folder to contain image files pertaining to
        the class that the folder represents.
        """
        content = os.listdir(self.datapath)
        classes = []
        total_images = 0

        # Check that each dir contains only images
        for directory in content:
            if self.verify_class(directory) is True:
                classes.append(directory)
                num_images = len(
                    os.listdir(os.path.join(self.datapath, directory)))
                self.num_instances[directory] = num_images
                total_images += num_images

        self.classes = classes
        self.total_images = total_images

    def verify_class(self, directory):
        """Verifies that class directory contains
        all images only. Only supports JPG and PNG
        images.
        """
        valid_image_ext = ['.jpg', '.jpeg', '.png']
        files = os.listdir(os.path.join(self.datapath, directory))

        if len(files) > 0:
            for file in files:
                _, ext = os.path.splitext(file)
                if ext not in valid_image_ext:
                    return False
            return True
        else:
            return False

    def rename_images(self):
        """Helper function to rename all images inside a directory to
        a standard naming scheme.
        """
        for label in self.classes:
            path = os.path.join(self.datapath, label)
            for i, image in enumerate(os.listdir(path)):
                _, ext = os.path.splitext(image)
                new_name = label + str(i) + ext
                src = os.path.join(path, image)
                dst = os.path.join(path, new_name)
                os.rename(src, dst)
            print('Finished renaming images for class {}'.format(label))

    def split_train_test_val(self, test_size=0.2, stratified=True):
        """Splits each valid folder of class images into
        train/val/test directories.2
        """
        train_dir = os.path.join(self.datapath, 'train')
        validation_dir = os.path.join(self.datapath, 'validation')
        test_dir = os.path.join(self.datapath, 'test')

        # Make directories for train/val/test
        for item in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(self.datapath, item), exist_ok=True)

        # Split train/val/test for each class
        for label in self.classes:
            data_path = os.path.join(self.datapath, label)
            num_examples = len(os.listdir(data_path))

            # Make the directories
            for directory in [train_dir, validation_dir, test_dir]:
                os.makedirs(os.path.join(directory, label), exist_ok=True)

            assert len(os.listdir(os.path.join(train_dir, label))) == 0, \
                'Train directory for {} is not empty.'.format(label)
            assert len(os.listdir(os.path.join(validation_dir, label))) == 0, \
                'Validation directory for {} is not empty'.format(label)
            assert len(os.listdir(os.path.join(test_dir, label))) == 0, \
                'Test directory for {} is not empty.'.format(label)

            # Shuffle the data
            datafiles = os.listdir(data_path)
            random.shuffle(datafiles)

            # Train-test split
            num_test = math.floor(num_examples*0.2)
            num_train = num_examples - num_test

            # Train-validation split
            num_val = math.floor(num_train*0.2)
            num_train = num_train - num_val

            # Train, validation, and test data
            train_files = datafiles[:num_train]
            val_files = datafiles[num_train:num_train+num_val]
            test_files = datafiles[num_train+num_val:]

            # Copy training data
            train_dir_class = os.path.join(train_dir, label)
            for filename in train_files:
                src = os.path.join(data_path, filename)
                dst = os.path.join(train_dir_class, filename)
                shutil.copyfile(src, dst)

            # Copy validation data
            val_dir_class = os.path.join(validation_dir, label)
            for filename in val_files:
                src = os.path.join(data_path, filename)
                dst = os.path.join(val_dir_class, filename)
                shutil.copyfile(src, dst)

            # Copy test data
            test_dir_class = os.path.join(test_dir, label)
            for filename in test_files:
                src = os.path.join(data_path, filename)
                dst = os.path.join(test_dir_class, filename)
                shutil.copyfile(src, dst)

            # Store number of instances
            self.num_train[label] = len(os.listdir(train_dir_class))
            self.num_val[label] = len(os.listdir(val_dir_class))
            self.num_test[label] = len(os.listdir(test_dir_class))

            print('Completed train/val/test split for class {}'.format(label))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True)
    args = vars(ap.parse_args())

    # Create dataset preparer object
    dataset = DatasetPreparer(args['dataset'])

    # Observe unique classes
    print('Unique classes: ', dataset.classes)

    # Split train-val-test
    dataset.split_train_test_val(test_size=0.2)
