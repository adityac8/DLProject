import os
from dataset_RGB import DataLoaderTrain3Im, DataLoaderVal3Im, DataLoaderTrain5Im, DataLoaderVal5Im, DataLoaderTest3Im, DataLoaderTest5Im

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if len(img_options['wb_settings'])==3:
        return DataLoaderTrain3Im(rgb_dir, img_options)
    else:
        return DataLoaderTrain5Im(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if len(img_options['wb_settings'])==3:
        return DataLoaderVal3Im(rgb_dir, img_options)
    else:
        return DataLoaderVal5Im(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    if len(img_options['wb_settings'])==3:
        return DataLoaderTest3Im(rgb_dir, img_options)
    else:
        return DataLoaderTest5Im(rgb_dir, img_options)
