# !/usr/bin/env python
# coding: utf-8

### Use this Jupyter Notebook as a guide to run your trained model in inference mode

# created by Anton Morgunov

# inspired by [tensorflow object detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)

# Your first step is going to specify which unit you are going to work with for inference. Select between GPU or CPU and follow the below instructions for implementation.
import json
import argparse
import os
# from tkinter import N
from pycocotools.coco import COCO
import matching_score_code as match
# Initialize parser
parser = argparse.ArgumentParser()
import csv
import random



import os # importing OS in order to make GPU visible
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# # specify which device you want to work on.
# # Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="0" # TODO: specify your computational device

import tensorflow as tf # import tensorflow

# # checking that GPU is found
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# # other import
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from training_pipeline.utils.validation import inference_with_plot

# # Next you will import import scripts that were already provided by Tensorflow API. **Make sure that Tensorflow is your current working directory.**

# # In[ ]:


import sys # importyng sys in order to access scripts located in a different folder


path2scripts = "/home/Tensorflow/models/research/" # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import


# # In[ ]:


# # importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder



if __name__ == "__main__":
    # Adding optional argument
    parser.add_argument("-c", "--path2config", help = "workspace/exported-models/<folder with the model of your choice>/pipeline.config",required=True)
    parser.add_argument("-m", "--path2model", help = "workspace/exported-models/<folder with the model of your choice>/",required=True)
    parser.add_argument("-l", "--path2labelmap", help = "path to lable map",required=True)
    parser.add_argument("-i", "--path2images", help = "path to lable map",required=True)
    parser.add_argument("-a", "--annotations", help = "path to annotations",required=True)
    parser.add_argument("-t", "--task", help = "task",required=True)
    parser.add_argument("-o", "--path2outdir", help = "Dir to save output images",required=True)

    # # Read arguments from command line
    args = parser.parse_args()

    path2config = os.path.abspath(args.path2config)
    path2model = os.path.abspath(args.path2model)
    path2label_map = os.path.abspath(args.path2labelmap)
    path2images = os.path.abspath(args.path2images)
    annotations = os.path.abspath(args.annotations)
    path2outdir = os.path.abspath(args.path2outdir)
    path2exported_folder = path2config.rpartition("/")[0]

    configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
    model_config = configs['model'] # recreating model config
    detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

    category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)
    print(f"Category_index: {category_index}")
    # # In[ ]:

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()
    imagelist = os.listdir(path2images)
    #results = inference_as_raw_output(imagelist, path2dir=path2images)
    inference_with_plot(detection_model=detection_model,
                        path2images=imagelist,
                        box_th=0.9,
                        path2dir=path2images,
                        annotations=annotations,
                        category_index = category_index,
                        path2outdir=path2outdir,
                        mytask="NE")
    # print(results)