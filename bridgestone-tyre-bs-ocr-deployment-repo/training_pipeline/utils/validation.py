# !/usr/bin/env python
# coding: utf-8

### Use this Jupyter Notebook as a guide to run your trained model in inference mode

# created by Anton Morgunov

# inspired by [tensorflow object detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)

# Your first step is going to specify which unit you are going to work with for inference. Select between GPU or CPU and follow the below instructions for implementation.
import json
# import argparse
import os
# from tkinter import N
from pycocotools.coco import COCO
import training_pipeline.matching_score_code as match
# Initialize parser

import csv
import random
# Adding optional argument

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
import cv2


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


# # Now you can import and build your trained model:

# # In[ ]:


# # NOTE: your current working directory should be Tensorflow.

# # In[ ]:


# # Next, path to label map should be provided. Category index will be created based on labal map file

# # In[ ]:
# # Now, a few supporting functions will be defined

# # In[ ]:


def detect_fn(detection_model, image):
    """
    Detect objects in image.
    
    Args:
      image: (tf.tensor): 4D input image
      
    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    
    return np.array(Image.open(path))


# # **Next function is the one that you can use to run inference and plot results an an input image:**

# # In[ ]:


def inference_with_plot(detection_model=None,
                        path2images=None, box_th=0.25,
                        path2dir=False,
                        annotations=False,
                        mytask=None,
                        category_index=None,
                        path2outdir=None):
    """
    Function that performs inference and plots resulting b-boxes
    
    Args:
      path2images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction.
      
    Returns:
      None
    """

    if mytask == "NE":
        stat = {'correct_percent':[], 'wrong_percent': [], 'total_correct': 0,\
                'total_wrong': 0, 'total': 0, '37': [0, 0, 0], '1': [0, 0, 0],\
                '2': [0, 0, 0],'3': [0, 0, 0], '4': [0, 0, 0], '5': [0, 0, 0], '6': [0, 0, 0],\
                '7': [0, 0, 0], '8': [0, 0, 0], '9': [0, 0, 0]}
    else:
        stat = {'correct_percent':[], 'wrong_percent': [], 'total_correct': 0,\
             'total_wrong': 0, 'total': 0, '1': [0, 0, 0], '2': [0, 0, 0]}
    coco_annotation = False
    if annotations:
        coco_annotation = COCO(annotations)
        img_ids = coco_annotation.getImgIds()
        ann_ids = coco_annotation.getAnnIds(img_ids)
        anns = coco_annotation.loadAnns(ann_ids)

        if coco_annotation:
            tyre_number_extractions = []
            for img_id in tqdm(img_ids):
                image_info = coco_annotation.loadImgs(img_id)[0]
                image_name = image_info["file_name"]
#                 #print(image_name)
                trimmed_image_name = image_name.rpartition("/")[-1]
                # allowed_images = ['sqr_cropped_img_0048023419_RMS.jpeg', 'sqr_cropped_img_0160400819_SNM.jpeg', \
                #                 'sqr_cropped_img_0263360519_RMS.jpeg', 'sqr_cropped_img_B252642918_RMS.jpeg', \
                #                 'sqr_cropped_img_B285030519_RMS.jpeg', 'sqr_cropped_img_C0226630519_SNM.jpeg', \
                #                 'sqr_cropped_img_0058751819_SNM.jpeg', 'sqr_cropped_img_0127881019_SNM.jpeg']

                # if trimmed_image_name not in allowed_images:
                    # continue

                image_id = image_info["id"]
                bboxes = [ann['bbox'] for ann in anns if not(ann['segmentation']) and ann['image_id'] == img_id]

                image_width = image_info['width']
                image_height = image_info['height']
                normalized_bboxes = [[float(bbox[0])/image_width, \
                                      float(bbox[1])/image_height, \
                                      float(bbox[0] + bbox[2])/image_width, \
                                      float(bbox[1] +bbox[3])/image_height] \
                                      for bbox in bboxes] #format: (x1,y1,x2,y2)
#                 #print(normalized_bboxes)
                cat_ids = [ann['category_id'] for ann in anns if not(ann['segmentation']) and ann['image_id'] == img_id]
                annotation_info = list(zip(normalized_bboxes, cat_ids))
#                 # print(annotation_info)
                avg_width = get_average_bbox_width(normalized_bboxes)
                #avg_width = avg_width*image_width

                image_path = os.path.join(path2dir, image_name.strip())
                image_np = load_image_into_numpy_array(image_path)

                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                detections = detect_fn(detection_model, input_tensor)

#                 # All outputs are batches tensors.
#                 # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#                 # We're only interested in the first num_detections.
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                              for key, value in detections.items()}

                detections['num_detections'] = num_detections

#                 # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()
                print(f'Image name: {image_name}')
                
                detections, stat = validate_output(annotation_info, detections, label_id_offset, box_th,\
                                                   stat, category_index)

                # master_tyre_number, predicted_string = get_predicted_tyre_number_string(detections, avg_width, image_name)
                # match_no_match = match.regexMatcher(master_tyre_number, predicted_string, 4)
                # tyre_number_extractions.append((master_tyre_number, predicted_string, match_no_match))
                
                # print(f"Master : {master_tyre_number} \n \
                #         Predicted: {predicted_string} \n \
                #         Match: {match_no_match}")
                viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=200,
                        min_score_thresh=box_th,
                        agnostic_mode=False,
                        line_thickness=7,
                        skip_scores=True)

                plt.figure(figsize=(15,10))
                # trimmed_image_name = image_name.rpartition("/")[-1]
                plt.imsave(os.path.join(path2outdir, trimmed_image_name), image_np_with_detections)
                print('Done')
        logs = []
        print(f"Stat: {stat}")
        logs.append(f"Stat: {stat}")
        bins = [-10, 0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        hist, edges = np.histogram(stat['correct_percent'], bins)
        freq = hist*100/float(hist.sum())
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(stat['correct_percent'], bins = bins,edgecolor = "black",label=freq, color='c')
        ax.set_ylabel('Number of images')
        ax.set_xlabel('Percentage')
        ax.set_title('Correct prediction distribution')
        rects = ax.patches
        labels = [ "{0:.2f}%".format(round(i,2)) for i in freq]
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                    ha='center', va='bottom')
        plt.savefig(os.path.join(path2outdir, "correct_prediction_distribution.jpeg"))
        no_preds = stat['correct_percent'].count(0.0)
        correct_average_including_zero_predictions = sum(stat['correct_percent'])/len(stat['correct_percent'])
        try:
            correct_average_without_zero = sum(stat['correct_percent'])/(len(stat['correct_percent'])-no_preds)
        except ZeroDivisionError:
            print("Getting ZeroDivisionError")
            correct_average_without_zero = 0.0
        print(f"correct_average_including_zero_predictions: {correct_average_including_zero_predictions}")
        logs.append(f"correct_average_including_zero_predictions: {correct_average_including_zero_predictions}")
        print(f"correct_average_without_zero: {correct_average_without_zero}")
        logs.append(f"correct_average_without_zero: {correct_average_without_zero}")
        print(f"Total characters: {stat['total']}")
        logs.append(f"Total characters: {stat['total']}")
        corr_percent = (stat['total_correct']/ stat['total'])*100
        # corr_percent_without_zero = (stat['total_correct']/ (stat['total']-no_preds))*100
        print(f"Total correct:  {stat['total_correct']}")
        logs.append(f"Total correct:  {stat['total_correct']}")
        print(f"Correct % :  {corr_percent}")
        logs.append(f"Correct % :  {corr_percent}")
        # print(f"Correct % without zero:  {corr_percent_without_zero}")

        hist, edges = np.histogram(stat['wrong_percent'], bins)
        freq = hist*100/float(hist.sum())
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(stat['wrong_percent'], bins = bins,edgecolor = "black",label=freq, color='c')
        ax.set_ylabel('Number of images')
        ax.set_xlabel('Percentage')
        ax.set_title('Wrong prediction distribution')
        rects = ax.patches
        labels = [ "{0:.2f}%".format(round(i,2)) for i in freq]
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                    ha='center', va='bottom')
        plt.savefig(os.path.join(path2outdir, "wrong_prediction_distribution.jpeg"))
        no_preds = stat['wrong_percent'].count(0.0)
        wrong_average_including_zero_predictions = sum(stat['wrong_percent'])/len(stat['wrong_percent'])
        print(f"Wrong_average_including_zero_predictions: {wrong_average_including_zero_predictions}")
        logs.append(f"Wrong_average_including_zero_predictions: {wrong_average_including_zero_predictions}")
        wrong_percent = (stat['total_wrong']/ stat['total'])*100
        print(f"Wrong % :  {wrong_percent}")
        logs.append(f"Wrong % :  {wrong_percent}")
        # if args.task == "NE":
        try:
            wrong_average_without_zero = sum(stat['wrong_percent'])/(len(stat['wrong_percent'])-no_preds)
        except ZeroDivisionError:
            print("Getting ZeroDivisionError")
            wrong_average_without_zero = 0.0
            print(f"Wrong_average_without_zero: {wrong_average_without_zero}")
            logs.append(f"Wrong_average_without_zero: {wrong_average_without_zero}")
            # wrong_percent_without_zero = (stat['total_wrong']/ (stat['total']-no_preds))*100
            print(f"Total wrong:  {stat['total_wrong']}")
            logs.append(f"Total wrong:  {stat['total_wrong']}")
            # print(f"Wrong % without zero:  {wrong_percent_without_zero}")
        
        if mytask == "NE":
            chars_stat = {'0': stat['37'], '1': stat['1'],'2': stat['2'],'3': stat['3'], \
                '4': stat['4'], '5': stat['5'], '6': stat['6'],\
                '7': stat['7'], '8': stat['8'], '9': stat['9']}
            labels = [0,1,2,3,4,5,6,7,8,9]
        else:
            chars_stat = {'1': stat['1'], '2': stat['2']}
            labels = ['char', 'alp']

        indi = list(zip(*chars_stat.values()))

        x = np.arange(0, 2* len(labels), 2)  # the label locations
        width = 0.55  # the width of the bars

        fig, ax = plt.subplots(figsize =(10, 7))
        rects1 = ax.bar(x, indi[0], width, label='Total appearance', color = 'b')
        rects2 = ax.bar(x + width, indi[1], width, label='Correctly predicted', color = 'g')
        rects3 = ax.bar(x + 2* width, indi[2], width, label='Wrongly predicted', color = 'r')

#       Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Count')
        ax.set_xlabel('Character')
        ax.set_title('Character (0-9) prediction results')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()
        plt.savefig(os.path.join(path2outdir,"char_stats.jpeg"))
        csv_file_path = os.path.join(path2outdir, "data.csv")
        with open(csv_file_path, "wt") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(["Master", "Predicted", "Match"])
            writer.writerows(tyre_number_extractions)

        joined_log_string = '\n'.join(logs)
        stat_txt_file_path = os.path.join(path2outdir, "stat_text.txt")
        with open(stat_txt_file_path, "w") as file:
            file.write(joined_log_string)

        print("Completed")



def inference_with_plot_production(detection_model=None,
                        input_image=None, 
                        box_th=0.25,
                        mytask=None,
                        category_index=None,
                        viz_flag=True):

    input_tensor = tf.convert_to_tensor(np.expand_dims(input_image, 0), dtype=tf.float32)
    detections = detect_fn(detection_model, input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = input_image.copy()
    
    detections = suppress_tag_label(detections, label_id_offset, category_index)
    output_info = list(zip(detections['detection_boxes'],
                           detections['detection_scores'],
                           detections['detection_classes'] + label_id_offset
                          )
                      )
    boxes, scores, classes = nms(output_info, max_prob=0.0)
    boxes, scores, classes = remove_low_confidence_boxes(boxes, scores, classes, box_th)

    detections['detection_boxes'] = np.array(boxes) # format: [y1, x1, y2, x2]
    detections['detection_scores'] = np.array(scores)
    detections['detection_classes'] = np.array(classes)

    if viz_flag:
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=box_th,
                agnostic_mode=False,
                line_thickness=7,
                skip_scores=True)
        print(image_np_with_detections.shape)
        image_np_with_detections = get_crop_of_tag(image_np_with_detections, detections)
    return image_np_with_detections, detections


def get_some_stats(tag_number, cutoff_dict):
    if tag_number == "no_detection":
        cutoff_dict["equal_to_0"] += 1
        return cutoff_dict
    else:
        num_of_spaces = tag_number.count("-")
        num_of_chars_extracted = len(tag_number)-num_of_spaces
    if num_of_chars_extracted == 0:
        cutoff_dict["equal_to_0"] += 1
    elif num_of_chars_extracted <= 4:
        cutoff_dict["less_than_equal_to_4"] += 1
    elif num_of_chars_extracted > 4:
        cutoff_dict["greater_than_4"] += 1

    return cutoff_dict


def element_by_element_matching_where_len_is_same(tag_number_gt, tag_number_pred, N=5):
    no_of_detected_chars = len(tag_number_pred)-tag_number_pred.count('-')
    if no_of_detected_chars < N:
        return "very_few_chars_extracted"
    non_matching_chars = 0
    if tag_number_gt == tag_number_pred:
        return "full_match"

    for i in range(len(tag_number_gt)):
        if tag_number_gt[i] == tag_number_pred[i] or tag_number_pred[i] == "-":
            pass    
        else:
            non_matching_chars += 1
    if non_matching_chars == 0:
        return "full_match_with_spaces"
    elif non_matching_chars == 1:
        return "partial_match"
    else:
        return False


def inference_of_whole_pipeline(path2images=None,
                        annotations=False,
                        path2outdir=None,
                        Extractor=None,
                        box_th=0.9,
                        mytask="NE",
                        category_index= None):
    """
    """

    if mytask == "NE":
        stat = {'correct_percent':[], 'wrong_percent': [], 'total_correct': 0,\
                'total_wrong': 0, 'total': 0, '37': [0, 0, 0], '1': [0, 0, 0],\
                '2': [0, 0, 0],'3': [0, 0, 0], '4': [0, 0, 0], '5': [0, 0, 0], '6': [0, 0, 0],\
                '7': [0, 0, 0], '8': [0, 0, 0], '9': [0, 0, 0]}
    else:
        stat = {'correct_percent':[], 'wrong_percent': [], 'total_correct': 0,\
             'total_wrong': 0, 'total': 0, '1': [0, 0, 0], '2': [0, 0, 0]}
    coco_annotation = False
    if annotations:
        coco_annotation = COCO(annotations)
        img_ids = coco_annotation.getImgIds()
        ann_ids = coco_annotation.getAnnIds(img_ids)
        anns = coco_annotation.loadAnns(ann_ids)
        total_image_count = len(img_ids)


        if coco_annotation:
            tyre_number_extractions = []
            ocr_output = []
            count_where_in_and_out_tag_len_is_same = 0
            count_where_in_and_out_tag_len_is_same_but_very_few_chars_extracted = 0
            count_where_in_and_out_tag_len_is_not_the_same = 0
            count_where_element_by_element_fully_match_with_spaces = 0
            count_where_element_by_element_fully_match = 0
            count_where_len_is_same_and_ele_by_ele_do_not_fully_match = 0
            count_where_len_is_same_and_ele_by_ele_do_not_match_fully_or_partially = 0
            count_where_in_and_out_tag_len_is_not_the_same_and_very_few_chars_extracted = 0
            count_where_element_by_element_partial_match = 0
            count_where_regex_match_and_not_element_by_element = 0
            not_a_match = 0
            is_a_match = ""
            out = ""
            # total_image_count=0
            cutoff_dict = {"less_than_equal_to_4" : 0, "greater_than_4": 0, "equal_to_0": 0}
            logs = []
            for img_id in tqdm(img_ids):
                if img_id == 5:
                    break
                image_info = coco_annotation.loadImgs(img_id)[0]
                image_name = image_info["file_name"]
                trimmed_image_name = image_name.rpartition("/")[-1]
                print("trimmed_image_name")
                print(trimmed_image_name)
                name = trimmed_image_name.split(".j")[0]
                another_split_list = name.split('_')
                tag_number_gt = another_split_list[0]

                bboxes = [ann['bbox'] for ann in anns if not(ann['segmentation']) and ann['image_id'] == img_id]
                image_path = os.path.join(path2images, image_name.strip())
                img_pic = cv2.imread(image_path)
                image_width = image_info['width']
                image_height = image_info['height']
                # for box in bboxes:
                #     cv2.rectangle(img_pic, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,0,0), 2)
                # plt.imsave(os.path.join(path2outdir, "ann_plotted_"+trimmed_image_name), img_pic)
                # break
                normalized_bboxes = [[float(bbox[0])/image_width, \
                                      float(bbox[1])/image_height, \
                                      float(bbox[0] + bbox[2])/image_width, \
                                      float(bbox[1] +bbox[3])/image_height] \
                                      for bbox in bboxes] #format: (x1,y1,x2,y2)
#                 #print(normalized_bboxes)

                cat_ids = [ann['category_id'] for ann in anns if not(ann['segmentation']) and ann['image_id'] == img_id]
                
                annotation_info = list(zip(normalized_bboxes, cat_ids))
                avg_width = get_average_bbox_width(normalized_bboxes)

                avg_width = avg_width*image_width

                try:
                    image, mask_image, tag_number, detections_NE, corr_x, corr_y, sqr_img_w, \
                                                        sqr_img_h, rotate_matrix = Extractor.run(image_path, img_id)
                
                except:
                    print(f"Error in file name {image_name}")  
                #When we have pickled data ready
                # try:
                #     tag_number, detections_NE = Extractor.run(image_path, img_id)
                #     print(f"Predicted tag number: {tag_number}")
                # except:
                #     print(f"Error in file name {image_name}")
                #     continue

                if len(tag_number_gt) == len(tag_number):
                    count_where_in_and_out_tag_len_is_same += 1
                    if element_by_element_matching_where_len_is_same(tag_number_gt, tag_number, N=5) == "full_match":
                        print("Match successful")
                        count_where_element_by_element_fully_match += 1
                        is_a_match = 1
                    elif element_by_element_matching_where_len_is_same(tag_number_gt, tag_number, N=5) == "full_match_with_spaces":
                        print("Match successful")
                        count_where_element_by_element_fully_match_with_spaces += 1
                        is_a_match = 1
                    elif element_by_element_matching_where_len_is_same(tag_number_gt, tag_number, N=5) == "partial_match":
                        count_where_len_is_same_and_ele_by_ele_do_not_fully_match += 1
                        print("Match successful")
                        count_where_element_by_element_partial_match += 1
                        is_a_match = 1
                    elif element_by_element_matching_where_len_is_same(tag_number_gt, tag_number, N=5) == "very_few_chars_extracted":
                        print("Match unsuccessful")
                        is_a_match = 0
                        not_a_match += 1
                        count_where_in_and_out_tag_len_is_same_but_very_few_chars_extracted += 1
                    else:
                        count_where_len_is_same_and_ele_by_ele_do_not_fully_match += 1
                        count_where_len_is_same_and_ele_by_ele_do_not_match_fully_or_partially +=1
                        not_a_match += 1
                        is_a_match = 0
                        print("Match unsuccessful")
                else:
                    count_where_in_and_out_tag_len_is_not_the_same += 1
                    no_of_detected_chars = len(tag_number)-tag_number.count('-')
                    if no_of_detected_chars < 5:
                        count_where_in_and_out_tag_len_is_not_the_same_and_very_few_chars_extracted += 1
                        not_a_match += 1
                    else:
                        if match.regexMatcher(tag_number_gt, tag_number, 5):
                            count_where_regex_match_and_not_element_by_element += 1
                            is_a_match = 1
                            print("Regex successful")
                            logs.append(f"tag_number_gt where regex got successful: {tag_number_gt}")
                            logs.append(f"tag_number where regex got successful: {tag_number}")
                        else:
                            not_a_match += 1
                            is_a_match = 0
                            print("Regex unsuccessful")

                match_no_match = match.regexMatcher(tag_number_gt, tag_number, 5)
                if match_no_match:
                    out = 'Yes'
                else:
                    out = 'No'
                ocr_output.append({"FileName": image_name, "tyre_OCR": tag_number, \
                                "Match/No_Match": out, 'new_match?': is_a_match})
                                    
                cutoff_dict = get_some_stats(tag_number, cutoff_dict)

                # print(rotate_matrix.shape)
                # inv_rotation_matrix = np.transpose(rotate_matrix)
                # print(rotate_matrix.shape)
                # for bbox in normalized_bboxes:
                #     print("bbox")
                #     print(bbox)
                #     x_y = np.array([[bbox[0], bbox[1]]])
                #     x_y = cv2.transform(x_y, rotate_matrix)
                #     print("x_y")
                #     print(x_y)
                
                # for bboxes in detections_NE["detection_boxes"]:
                #     # format [y1, x1, y2, x2]
                #     bboxes[0] = ((bboxes[0]*sqr_img_h) + corr_y)#/image_height
                #     bboxes[1] = ((bboxes[1]*sqr_img_w) + corr_x)#/image_width
                #     bboxes[2] = ((bboxes[2]*sqr_img_h) + corr_y)#/image_height
                #     bboxes[3] = ((bboxes[3]*sqr_img_w) + corr_x)#/image_width
                #     x_y = np.array([[[bboxes[1], bboxes[0]]], [[bboxes[3], bboxes[2]]]])
                #     x_y = cv2.transform(x_y, inv_rotation_matrix)

                    # bboxes[1] = x_y[0][0][0]
                    # bboxes[0] = x_y[0][0][1]
                    # bboxes[3] = x_y[1][0][0]
                    # bboxes[2] = x_y[1][0][1]
                    # print("x_y")
                    # print(x_y)
                    # a = bboxes[0]*image_height
                    # b = bboxes[1]*image_width
                    # c = bboxes[2]*image_height
                    # d = bboxes[3]*image_width
                    # print(bboxes[0])
                    # print(bboxes[1])
                    # print(bboxes[2])
                    # print(bboxes[3])
                    # cv2.rectangle(img_pic, (bboxes[1], bboxes[0]), (bboxes[3], bboxes[2]), (0,0,255), 2)
                    # cv2.rectangle(img_pic, (b, a), (d, c), (255,0,255), 5)
                # plt.imsave(os.path.join(path2outdir, "plotted"+trimmed_image_name), img_pic)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()   
#                 # All outputs are batches tensors.
#                 # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#                 # We're only interested in the first num_detections.
                # num_detections = int(detections.pop('num_detections'))
                # detections = {key: value[0, :num_detections].numpy()
                            #   for key, value in detections.items()}

                # detections['num_detections'] = num_detections

#                 # detection_classes should be ints.
                # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                # label_id_offset = 0
                # image_np_with_detections = image_np.copy()
                # print(f'Image name: {image_name}')
                # print("annotation_info")
                # print(annotation_info)
                # print("before validate")
                # print(detections_NE["detection_boxes"])
                # print(detections_NE["detection_scores"])
                # print(detections_NE["detection_classes"])
                # detections, stat = validate_output(annotation_info, detections_NE, label_id_offset, box_th,\
                #                                    stat, category_index)
                # print("after validate")
                # print(detections["detection_boxes"])
                # print(detections["detection_scores"])
                # print(detections["detection_classes"])
                # tyre_number_extractions.append((master_tyre_number, predicted_string, match_no_match))
                
                # print(f"Master : {master_tyre_number} \n \
                #         Predicted: {predicted_string} \n \
                #         Match: {match_no_match}")
                # viz_utils.visualize_boxes_and_labels_on_image_array(
                #         image_np_with_detections,
                #         detections['detection_boxes'],
                #         detections['detection_classes'],
                #         detections['detection_scores'],
                #         category_index,
                #         use_normalized_coordinates=True,
                #         max_boxes_to_draw=200,
                #         min_score_thresh=box_th,
                #         agnostic_mode=False,
                #         line_thickness=7,
                #         skip_scores=True)

                plt.figure(figsize=(15,10))
                plt.imsave(os.path.join(path2outdir, trimmed_image_name), mask_image)
                print('Done')
            fieldnames = ['FileName', 'tyre_OCR', 'Match/No_Match', 'new_match?']
            with open('/home/host/output/chi_ocr.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ocr_output)
#         print(f"Stat: {stat}")
        logs.append(f"cutoff_dict: {cutoff_dict}")
        logs.append(f"Total number of images: {total_image_count}")
        logs.append(f"Number of images where the length of tag output and tag number is the same: {count_where_in_and_out_tag_len_is_same}")
        logs.append(f"Number of images where the length of tag output and tag number is NOT the same: {count_where_in_and_out_tag_len_is_not_the_same}")
        logs.append(f"Number of images where element by element 100% match: {count_where_element_by_element_fully_match}")
        logs.append(f"Number of images where element by element fully match with spaces: {count_where_element_by_element_fully_match_with_spaces}")
        logs.append(f"Number of images where output len is same but very few chars extracted: {count_where_in_and_out_tag_len_is_same_but_very_few_chars_extracted}")
        logs.append(f"Number of images where output len is same but ele by ele do not fully match: {count_where_len_is_same_and_ele_by_ele_do_not_fully_match}")
        logs.append(f"Number of images where output len is same but ele by ele neither match fully nor partially: {count_where_len_is_same_and_ele_by_ele_do_not_match_fully_or_partially}")
        logs.append(f"Number of images where output len is different but very few chars extracted : {count_where_in_and_out_tag_len_is_not_the_same_and_very_few_chars_extracted}")
        logs.append(f"Number of images where element by element partially match: {count_where_element_by_element_partial_match}")
        logs.append(f"Number of images where there is a regex match: {count_where_regex_match_and_not_element_by_element}")
        logs.append(f"Total number of images where there is a NO match: {not_a_match}")
        
        total_matches = count_where_element_by_element_fully_match + \
                        count_where_element_by_element_fully_match_with_spaces + \
                        count_where_element_by_element_partial_match + \
                        count_where_regex_match_and_not_element_by_element
        logs.append(f"Total number of matches: {total_matches}")
        accuracy = (total_matches/(total_matches+not_a_match))*100
        logs.append(f"% Accuracy: {accuracy} %")


#         bins = [-10, 0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#         hist, edges = np.histogram(stat['correct_percent'], bins)
#         freq = hist*100/float(hist.sum())
#         fig, ax = plt.subplots(figsize =(10, 7))
#         ax.hist(stat['correct_percent'], bins = bins,edgecolor = "black",label=freq, color='c')
#         ax.set_ylabel('Number of images')
#         ax.set_xlabel('Percentage')
#         ax.set_title('Correct prediction distribution')
#         rects = ax.patches
#         labels = [ "{0:.2f}%".format(round(i,2)) for i in freq]
#         for rect, label in zip(rects, labels):
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
#                     ha='center', va='bottom')
#         plt.savefig(os.path.join(path2outdir, "correct_prediction_distribution.jpeg"))
#         no_preds = stat['correct_percent'].count(0.0)
#         correct_average_including_zero_predictions = sum(stat['correct_percent'])/len(stat['correct_percent'])
#         try:
#             correct_average_without_zero = sum(stat['correct_percent'])/(len(stat['correct_percent'])-no_preds)
#         except ZeroDivisionError:
#             print("Getting ZeroDivisionError")
#             correct_average_without_zero = 0.0
#         print(f"correct_average_including_zero_predictions: {correct_average_including_zero_predictions}")
#         logs.append(f"correct_average_including_zero_predictions: {correct_average_including_zero_predictions}")
#         print(f"correct_average_without_zero: {correct_average_without_zero}")
#         logs.append(f"correct_average_without_zero: {correct_average_without_zero}")
#         print(f"Total characters: {stat['total']}")
#         logs.append(f"Total characters: {stat['total']}")
#         corr_percent = (stat['total_correct']/ stat['total'])*100
#         # corr_percent_without_zero = (stat['total_correct']/ (stat['total']-no_preds))*100
#         print(f"Total correct:  {stat['total_correct']}")
#         logs.append(f"Total correct:  {stat['total_correct']}")
#         print(f"Correct % :  {corr_percent}")
#         logs.append(f"Correct % :  {corr_percent}")
#         # print(f"Correct % without zero:  {corr_percent_without_zero}")

#         hist, edges = np.histogram(stat['wrong_percent'], bins)
#         freq = hist*100/float(hist.sum())
#         fig, ax = plt.subplots(figsize =(10, 7))
#         ax.hist(stat['wrong_percent'], bins = bins,edgecolor = "black",label=freq, color='c')
#         ax.set_ylabel('Number of images')
#         ax.set_xlabel('Percentage')
#         ax.set_title('Wrong prediction distribution')
#         rects = ax.patches
#         labels = [ "{0:.2f}%".format(round(i,2)) for i in freq]
#         for rect, label in zip(rects, labels):
#             height = rect.get_height()
#             ax.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
#                     ha='center', va='bottom')
#         plt.savefig(os.path.join(path2outdir, "wrong_prediction_distribution.jpeg"))
#         no_preds = stat['wrong_percent'].count(0.0)
#         wrong_average_including_zero_predictions = sum(stat['wrong_percent'])/len(stat['wrong_percent'])
#         print(f"Wrong_average_including_zero_predictions: {wrong_average_including_zero_predictions}")
#         logs.append(f"Wrong_average_including_zero_predictions: {wrong_average_including_zero_predictions}")
#         wrong_percent = (stat['total_wrong']/ stat['total'])*100
#         print(f"Wrong % :  {wrong_percent}")
#         logs.append(f"Wrong % :  {wrong_percent}")
#         # if args.task == "NE":
#         try:
#             wrong_average_without_zero = sum(stat['wrong_percent'])/(len(stat['wrong_percent'])-no_preds)
#         except ZeroDivisionError:
#             print("Getting ZeroDivisionError")
#             wrong_average_without_zero = 0.0
#             print(f"Wrong_average_without_zero: {wrong_average_without_zero}")
#             logs.append(f"Wrong_average_without_zero: {wrong_average_without_zero}")
#             # wrong_percent_without_zero = (stat['total_wrong']/ (stat['total']-no_preds))*100
#             print(f"Total wrong:  {stat['total_wrong']}")
#             logs.append(f"Total wrong:  {stat['total_wrong']}")
#             # print(f"Wrong % without zero:  {wrong_percent_without_zero}")
        
#         if mytask == "NE":
#             chars_stat = {'0': stat['37'], '1': stat['1'],'2': stat['2'],'3': stat['3'], \
#                 '4': stat['4'], '5': stat['5'], '6': stat['6'],\
#                 '7': stat['7'], '8': stat['8'], '9': stat['9']}
#             labels = [0,1,2,3,4,5,6,7,8,9]
#         else:
#             chars_stat = {'1': stat['1'], '2': stat['2']}
#             labels = ['char', 'alp']

#         indi = list(zip(*chars_stat.values()))

#         x = np.arange(0, 2* len(labels), 2)  # the label locations
#         width = 0.55  # the width of the bars

#         fig, ax = plt.subplots(figsize =(10, 7))
#         rects1 = ax.bar(x, indi[0], width, label='Total appearance', color = 'b')
#         rects2 = ax.bar(x + width, indi[1], width, label='Correctly predicted', color = 'g')
#         rects3 = ax.bar(x + 2* width, indi[2], width, label='Wrongly predicted', color = 'r')

# #         # Add some text for labels, title and custom x-axis tick labels, etc.
#         ax.set_ylabel('Count')
#         ax.set_xlabel('Character')
#         ax.set_title('Character (0-9) prediction results')
#         ax.set_xticks(x, labels)
#         ax.legend()

#         ax.bar_label(rects1, padding=3)
#         ax.bar_label(rects2, padding=3)
#         ax.bar_label(rects3, padding=3)

#         fig.tight_layout()
#         plt.savefig(os.path.join(path2outdir,"char_stats.jpeg"))
#         csv_file_path = os.path.join(path2outdir, "data.csv")
#         with open(csv_file_path, "wt") as fp:
#             writer = csv.writer(fp, delimiter=",")
#             writer.writerow(["Master", "Predicted", "Match"])
#             writer.writerows(tyre_number_extractions)

        joined_log_string = '\n'.join(logs)
        stat_txt_file_path = os.path.join(path2outdir, "stat_text.txt")
        with open(stat_txt_file_path, "w") as file:
            file.write(joined_log_string)

        print("Completed")


def get_crop_of_tag(image_np_with_detections, detections):
    y = 0
    H = 0
    height, width, C = image_np_with_detections.shape
    for bbox_pred in detections['detection_boxes']:
        # bbox_pred format: [y1, x1, y2, x2]
        x1 = bbox_pred[1]
        y1 = bbox_pred[0] 
        x2 = bbox_pred[3]
        y2 = bbox_pred[2]
        y = int(y1*height)
        H = int(np.abs((y2-y1) * height))
        print(bbox_pred, y, H)
        break
    
    tag_crop = None
    if y and H:
        tag_crop = image_np_with_detections[y-30:y+H+30,:]
    #print(tag_crop.shape)
    return tag_crop


# # Next, we will define a few other supporting functions:

# # In[ ]:

def get_average_bbox_width(norm_bboxes):
    #norm_bboxes is in the form [[x1,y1,x2,y2], [...], ...]
    width_list = [bbox[2]-bbox[0] for bbox in norm_bboxes]
    print(width_list)
    average_width = sum(width_list)/len(width_list)
    print(average_width)
    return average_width


def get_predicted_tyre_number_string(detections, avg_width, category_index):
    z = list(zip(detections['detection_boxes'], detections['detection_classes']))
    sorted_z = sorted(z, key=lambda x: x[0][3])  #boxes sorted by x2
    l_s = []
    box_list = []
    prev_x2 = 0
    gap_threshold = avg_width
    for i in range(len(sorted_z)):
        new_x1 = sorted_z[i][0][1]
        new_x2 = sorted_z[i][0][3]
        # box_list 
        cate_id = sorted_z[i][1]
        print("category_index")
        print(category_index)
        print("cate_id")
        print(cate_id)
        # print()
        cat_name = category_index[cate_id]['name']
        if cat_name == "alp":
            cat_name = "A"
        elif cat_name == "char":
            cat_name = "C"
        else:
            pass

        missed_char = '-'
        if i != 0:
            if new_x1-prev_x2 > gap_threshold and new_x1-prev_x2 < 2*gap_threshold:
                l_s.append(missed_char)
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 2*gap_threshold and new_x1-prev_x2 < 3*gap_threshold:
                for i in range(2):
                    l_s.append(missed_char)
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 3*gap_threshold and new_x1-prev_x2 < 4*gap_threshold:
                for i in range(3):
                    l_s.append(missed_char)
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 4*gap_threshold and new_x1-prev_x2 < 5*gap_threshold:
                for i in range(4):
                    l_s.append(missed_char)
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 5*gap_threshold and new_x1-prev_x2 < 6*gap_threshold:
                for i in range(5):
                    l_s.append(missed_char) 
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 6*gap_threshold and new_x1-prev_x2 < 7*gap_threshold:
                for i in range(6):
                    l_s.append(missed_char) 
                l_s.append(cat_name)
            elif new_x1-prev_x2 >= 7*gap_threshold and new_x1-prev_x2 < 8*gap_threshold:
                for i in range(7):
                    l_s.append(missed_char) 
                l_s.append(cat_name)
            else:
                l_s.append(cat_name)
        else:
            l_s.append(cat_name)
        prev_x2 = new_x2
    
    pred_s = "".join(l_s)
    return pred_s


def get_box(px2, i, gap_th, py1, py2):
    y1 = py1
    y2 = py2
    x1 = px2 + gap_th*i + gap_th*0.015
    x2 = x1 + gap_th + gap_th*0.005
    return (y1, x1, y2, x2)


def get_predicted_tyre_number_string_v2(detections, avg_gap_between_boxes, category_index):
    # detections['detection_boxes'] are of the form : y1, x1, y2, x2
    # z = list(zip(detections['detection_boxes'], detections['detection_classes']))
    sorted_z = sorted(detections, key=lambda x: x[2])  #boxes sorted by x2
    print(sorted_z)
    print(sorted_z[0].shape)
    l_s = []
    box_list = []   
    prev_x2 = 0
    gap_threshold = avg_gap_between_boxes
    for j in range(len(sorted_z)):
        new_x1 = sorted_z[j][0]
        new_x2 = sorted_z[j][2]
        # box_list
        bbox = sorted_z[j][:4]
        cate_id = sorted_z[j][5]
        cate_id = cate_id.item()
        print(cate_id)
        cat_name = category_index[cate_id]['name']
        if cat_name == "alp":
            cat_name = "A"
        elif cat_name == "char":
            cat_name = "C"
        else:
            pass

        missed_char = '-'
        if j != 0:
            diff = new_x1-prev_x2
            if diff >= gap_threshold and diff < 2*gap_threshold:
                l_s.append(missed_char)
                box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 2*gap_threshold and diff < 3*gap_threshold:
                for i in range(2):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 3*gap_threshold and diff < 4*gap_threshold:
                for i in range(3):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 4*gap_threshold and diff < 5*gap_threshold:
                for i in range(4):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 5*gap_threshold and diff < 6*gap_threshold:
                for i in range(5):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 6*gap_threshold and diff < 7*gap_threshold:
                for i in range(6):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            elif diff >= 7*gap_threshold and diff < 8*gap_threshold:
                for i in range(7):
                    l_s.append(missed_char)
                    box_list.append(get_box(prev_x2, 0, gap_threshold, prev_y1, prev_y2))
                l_s.append(cat_name)
                box_list.append(bbox)
            else:
                l_s.append(cat_name)
                box_list.append(bbox)
        else:
            l_s.append(cat_name)
            print(cat_name)
            box_list.append(bbox)
        # prev_x1 = new_x1
        prev_x2 = new_x2
        prev_y1 = sorted_z[j][1]
        prev_y2 = sorted_z[j][3]
        
    
    pred_s = "".join(l_s)
    return pred_s, box_list


# def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE):
#     tag_number = ""
#     if len(pred_string_CR):
#         for c, bbox_CR in zip(pred_string_CR, pred_box_list_CR):
#             # print()
#             x1 = bbox_CR[1]
#             y1 = bbox_CR[0] 
#             x2 = bbox_CR[3]
#             y2 = bbox_CR[2]
#             bbox_CR = [x1, y1, x2, y2]
#             # if c == "_":
#             #     tag_number += "_"
#             #     continue
#             match = 0
#             for bbox_NE, pred_class in zip(detections_NE["detection_boxes"], detections_NE["detection_classes"]):
#                 x1 = bbox_NE[1]
#                 y1 = bbox_NE[0] 
#                 x2 = bbox_NE[3]
#                 y2 = bbox_NE[2]
#                 bbox_NE = [x1, y1, x2, y2]
#                 overlap = intersection(bbox_CR, bbox_NE)/ min(square(bbox_CR), square(bbox_NE))
#                 if overlap >= 0.8:
#                     tag_number += category_index_NE[pred_class]['name']
#                     match = 1
#                     break
#             if match == 0:
#                 tag_number += "_"

#     return tag_number


# def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE, \
#                                     average_width):
#     tag_number = ""
#     # detections['detection_boxes'] are of the form : y1, x1, y2, x2
#     sorted_cr = sorted(pred_box_list_CR, key=lambda x: x[1])  #boxes sorted by x1
#     z = list(zip(detections_NE['detection_boxes'], detections_NE['detection_classes']))
#     sorted_ne = sorted(z, key=lambda x: x[0][1])  #boxes sorted by x1
#     # sorted_ne_copy = sorted_ne.copy
#     ne_starts_from_coordinate = sorted_ne[0][0][1]
#     cr_starts_from_coordinate = sorted_cr[0][1]
#     if cr_starts_from_coordinate-ne_starts_from_coordinate > average_width//2:
#         first_pred = sorted_ne[0][1]
#         tag_number += category_index_NE[first_pred]['name']

#     if len(pred_string_CR):
#         for c, bbox_CR in zip(pred_string_CR, sorted_cr):
#             x1 = bbox_CR[1]
#             y1 = bbox_CR[0] 
#             x2 = bbox_CR[3]
#             y2 = bbox_CR[2]
#             bbox_CR = [x1, y1, x2, y2]
#             # if c == "_":
#             #     tag_number += "_"
#             #     continue
#             match = 0
#             for bbox_NE, pred_class in sorted_ne:
#                 x1 = bbox_NE[1]
#                 y1 = bbox_NE[0] 
#                 x2 = bbox_NE[3]
#                 y2 = bbox_NE[2]
#                 bbox_NE = [x1, y1, x2, y2]
#                 overlap = intersection(bbox_CR, bbox_NE)/ min(square(bbox_CR), square(bbox_NE))
#                 print(overlap)
#                 if overlap >= 0.75:
#                     # check_if_detection_is_in_between(bbox_CR, sorted_ne_copy)
#                     tag_number += category_index_NE[pred_class]['name']
#                     # sorted_ne_copy.remove([bbox_NE, pred_class])
#                     match = 1
#                     break
#             if match == 0:
#                 tag_number += "_"

#     ne_ends_at_coordinate = sorted_ne[-1][0][3]
#     cr_ends_at_coordinate = sorted_cr[-1][3]
#     if ne_ends_at_coordinate - cr_ends_at_coordinate > average_width/2:
#         last_pred = sorted_ne[-1][1]
#         tag_number += category_index_NE[last_pred]['name']

#     return tag_number


# def removearray(L, arr):
#     ind = 0
#     size = len(L)
#     while ind != size and not np.array_equal(L[ind],arr):
#         ind += 1
#     if ind != size:
#         L.pop(ind)
#         return L
#     else:
#         raise ValueError('array not found in list.')

# def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE, \
#                                     average_width):
#     tag_number = ""
#     # detections['detection_boxes'] are of the form : y1, x1, y2, x2
#     sorted_cr = sorted(pred_box_list_CR, key=lambda x: x[1])  #boxes sorted by x1
#     z = list(zip(detections_NE['detection_boxes'], detections_NE['detection_classes']))
#     sorted_ne = sorted(z, key=lambda x: x[0][1])  #boxes sorted by x1
#     sorted_ne_copy = sorted_ne.copy()
#     # ne_starts_from_coordinate = sorted_ne[0][0][1]
#     # cr_starts_from_coordinate = sorted_cr[0][1]
#     # if cr_starts_from_coordinate-ne_starts_from_coordinate > average_width//2:
#         # first_pred = sorted_ne[0][1]
#         # tag_number += category_index_NE[first_pred]['name']

#     # if len(pred_string_CR):
#     # i = 0
#     for c, bbox_CR in zip(pred_string_CR, sorted_cr):
#         # print("sorted_cr")
#         # print(sorted_cr)
#         # print("\n")
#         print("type")
#         print(type(bbox_CR))
#         x1_cr = bbox_CR[1]
#         y1_cr = bbox_CR[0] 
#         x2_cr = bbox_CR[3]
#         y2_cr = bbox_CR[2]
#         bbox_CR_ = [x1_cr, y1_cr, x2_cr, y2_cr]
#         # if c == "_":
#         #     tag_number += "_"
#         #     continue
#         match = 0
#         # print("sorted_ne_copy")       
#         # print(sorted_ne_copy)
#         # print("len(sorted_ne_copy) before loop")
#         # len_sne = len(sorted_ne_copy)
#         # print(len_sne)
#         # i+=1
#         # print(f"i: {i}")

#         for bbox_NE, pred_class in sorted_ne_copy:
#             # print("match before")
#             # print(match)
#             print("type")
#             print(type(bbox_NE))
#             print("sorted_ne_copy array")
#             print(sorted_ne_copy)
#             print("\n")
#             print("current box and class")
#             print((bbox_NE, pred_class))
#             print("\n")
#             x1_ne = bbox_NE[1]
#             print("x1_ne")
#             print(x1_ne)
#             y1_ne = bbox_NE[0] 
#             x2_ne = bbox_NE[3]
#             y2_ne = bbox_NE[2]
#             bbox_NE_ = [x1_ne, y1_ne, x2_ne, y2_ne]
#             # if x1_cr-x1_ne > average_width//2:
#             #     print(f"missed in cr but is in ne: adding: {category_index_NE[pred_class]['name']}")
#             #     tag_number += category_index_NE[pred_class]['name']
#             #     print(f"tagnumber: {tag_number}")
#             #     # print("[bbox_NE, pred_class]")
#             #     # print([bbox_NE, pred_class])
#             #     # print("\n")
#             #     sorted_ne_copy.remove((bbox_NE, pred_class))
#             #     print("after removal sorted_ne_copy array:")
#             #     print(sorted_ne_copy)
#             #     match += 1
#             #     # print(f"{[bbox_NE, pred_class]} removed")
#             #     # continue
#             overlap = intersection(bbox_CR_, bbox_NE_)/ min(square(bbox_CR_), square(bbox_NE_))
#             # overlap = intersection(bbox_CR, bbox_NE)/ min(square(bbox_CR), square(bbox_NE))
#             # print(overlap)
#             if overlap >= 0.75:
#                 # check_if_detection_is_in_between(bbox_CR, sorted_ne_copy)
#                 print(f"overlap more: adding: {category_index_NE[pred_class]['name']}")
#                 tag_number += category_index_NE[pred_class]['name']
#                 print(f"tagnumber: {tag_number}")
#                 # print("[bbox_NE, pred_class]")
#                 # print([bbox_NE, pred_class])
#                 # sorted_ne_copy.remove((bbox_NE, pred_class))
#                 sorted_ne_copy = removearray(sorted_ne_copy, (bbox_NE, pred_class))
#                 print("after deletion")
#                 print(sorted_ne_copy)
#                 match += 1
#                 break
#         # print("match after")
#         # print(match)
#         if match == 0:
#             tag_number += "_"
#             print("missed in ne: adding: _")
#             print(f"tagnumber: {tag_number}")
#         # print("sorted_ne_copy")
#         # print(sorted_ne_copy)
#         # print("\n")
#         # print("len(sorted_ne_copy) after loop")
#         # len_sne = len(sorted_ne_copy)
#         # print(len_sne)

#     ne_ends_at_coordinate = sorted_ne[-1][0][3]
#     cr_ends_at_coordinate = sorted_cr[-1][3]
#     if ne_ends_at_coordinate - cr_ends_at_coordinate > average_width/2:
#         last_pred = sorted_ne[-1][1]
#         tag_number += category_index_NE[last_pred]['name']

#     return tag_number


#final 
# def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE, \
#                                     average_width):
#     tag_number = ""
#     # detections['detection_boxes'] are of the form : y1, x1, y2, x2
#     sorted_cr = sorted(pred_box_list_CR, key=lambda x: x[1])  #boxes sorted by x1
#     z = list(zip(detections_NE['detection_boxes'], detections_NE['detection_classes']))
#     sorted_ne = sorted(z, key=lambda x: x[0][1])  #boxes sorted by x1
#     sorted_ne_copy = sorted_ne.copy()

#     temp = []
#     for c, bbox_CR in zip(pred_string_CR, sorted_cr):
#         x1_cr = bbox_CR[1]
#         y1_cr = bbox_CR[0] 
#         x2_cr = bbox_CR[3]
#         y2_cr = bbox_CR[2]
#         bbox_CR_ = [x1_cr, y1_cr, x2_cr, y2_cr]
#         match = 0
#         # print("sorted_ne_copy before loop")       
#         # print(sorted_ne_copy)
#         if temp:
#             print("\n")
#             print(f"temp: {temp}")
#             for (bbox_NE, pred_class) in temp:
#                 print(f"sorted_ne_copy: {sorted_ne_copy}")
#                 print("bbox, pred class")
#                 print((bbox_NE, pred_class))
#                 sorted_ne_copy.remove((bbox_NE, pred_class))
#             # print("after removal sorted_ne_copy:")
#             # print(sorted_ne_copy)
#             temp = []
#         for i in range(len(sorted_ne_copy)):
#             bbox_NE = sorted_ne_copy[i][0]
#             pred_class = sorted_ne_copy[i][1]
#             # print("\n")
#             # print("sorted_ne_copy array at start of every inner loop iter")
#             # print(sorted_ne_copy)
#             # print("\n")
#             # print(f"i: {i}")
#             # print("current box and class")
#             # print(sorted_ne_copy[i])
#             # print("\n")
#             x1_ne = bbox_NE[1]
#             y1_ne = bbox_NE[0]
#             x2_ne = bbox_NE[3]
#             y2_ne = bbox_NE[2]
#             bbox_NE_ = [x1_ne, y1_ne, x2_ne, y2_ne]
#             if x1_cr-x1_ne > average_width/2:
#                 # print(f"missed in cr but is in ne: adding: {category_index_NE[pred_class]['name']}")
#                 tag_number += category_index_NE[pred_class]['name']
#                 # print(f"tagnumber: {tag_number}")
#                 # print("\n")
#                 temp.append((bbox_NE, pred_class))
#                 # print("appending to temp")
#                 match = 1
#                 # print("\n")
#                 continue
#             overlap = intersection(bbox_CR_, bbox_NE_)/ min(square(bbox_CR_), square(bbox_NE_))
#             if overlap >= 0.75:
#             #   check_if_detection_is_in_between(bbox_CR, sorted_ne_copy)
#                 # print(f"overlap more: adding: {category_index_NE[pred_class]['name']}")
#                 tag_number += category_index_NE[pred_class]['name']
#                 # print(f"tagnumber: {tag_number}")
#                 temp.append((bbox_NE, pred_class))
#                 match = 1
#                 break
#         if match == 0:
#             tag_number += "-"
#             # print("missed in ne: adding: _")
#             # print(f"tagnumber: {tag_number}")
#             # print("\n")

#     ne_ends_at_coordinate = sorted_ne[-1][0][3]
#     cr_ends_at_coordinate = sorted_cr[-1][3]
#     if ne_ends_at_coordinate - cr_ends_at_coordinate > average_width/2:
#         last_pred = sorted_ne[-1][1]
#         tag_number += category_index_NE[last_pred]['name']

#     return tag_number


# final v2
def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE, \
                                    average_width):
    tag_number = ""
    sorted_cr = sorted(pred_box_list_CR, key=lambda x: x[0])  #boxes sorted by x1
    sorted_ne = sorted(detections_NE, key=lambda x: x[2])
    sorted_ne_copy = sorted_ne.copy()

    temp = []
    for c, bbox_CR_ in zip(pred_string_CR, sorted_cr):
        x1_cr = bbox_CR_[0]
        match = 0
        if temp:
            for info in temp:
                sorted_ne_copy.remove(info)
            temp = []
        for i in range(len(sorted_ne_copy)):
            bbox_NE_ = sorted_ne_copy[i][:4]
            pred_class = sorted_ne_copy[i][-1].item()
            x1_ne = bbox_NE_[0]
            if x1_cr-x1_ne > average_width/2:
                tag_number += category_index_NE[pred_class]['name']
                temp.append(sorted_ne_copy[i])
                match = 1
                continue
            overlap = intersection(bbox_CR_, bbox_NE_)/ min(square(bbox_CR_), square(bbox_NE_))
            if overlap >= 0.75:
                tag_number += category_index_NE[pred_class]['name']
                temp.append(sorted_ne_copy[i])
                match = 1
                break
        if match == 0:
            tag_number += "-"

    ne_ends_at_coordinate = sorted_ne[-1][2]
    cr_ends_at_coordinate = sorted_cr[-1][2]
    if ne_ends_at_coordinate - cr_ends_at_coordinate > average_width/2:
        last_pred = sorted_ne[-1][1]
        tag_number += category_index_NE[last_pred]['name']

    return tag_number

 

# def get_cross_validated_tag_number(pred_string_CR, pred_box_list_CR, detections_NE, category_index_NE, \
#                                     average_width):
#     tag_number = ""
#     # detections['detection_boxes'] are of the form : y1, x1, y2, x2
#     cr = list(zip(pred_box_list_CR, pred_string_CR))
#     sorted_cr = sorted(cr, key=lambda x: x[1])  #boxes sorted by x1
#     ne = list(zip(detections_NE['detection_boxes'], detections_NE['detection_classes']))
#     sorted_ne = sorted(ne, key=lambda x: x[0][1])  #boxes sorted by x1
#     # ne_starts_from_coordinate = sorted_ne[0][0][1]
#     cr_starts_from_coordinate = sorted_cr[0][0][1]
#     # if cr_starts_from_coordinate-ne_starts_from_coordinate > average_width//2:
#         # first_pred = sorted_ne[0][1]
#         # tag_number += category_index_NE[first_pred]['name']

#     if len(pred_string_CR):
#         for i in range(len(sorted_cr)):
#             match = 0
#             bbox_CR = sorted_cr[i][0]
#             c = sorted_cr[i][1]
#             x1_cr = bbox_CR[1]
#             y1_cr = bbox_CR[0] 
#             x2_cr = bbox_CR[3]
#             y2_cr = bbox_CR[2]
#             bbox_CR = [x1_cr, y1_cr, x2_cr, y2_cr]
#             # match = 0
#             # for bbox_NE, pred_class in sorted_ne:
#             bbox_NE = sorted_ne[i][0]
#             x1_ne = bbox_NE[1]
#             y1_ne = bbox_NE[0] 
#             x2_ne = bbox_NE[3]
#             y2_ne = bbox_NE[2]
#             bbox_NE = [x1_ne, y1_ne, x2_ne, y2_ne]
#             pred_class = sorted_ne[i][1]
#             overlap = intersection(bbox_CR, bbox_NE)/ min(square(bbox_CR), square(bbox_NE))
#             # print(overlap)
#             if overlap >= 0.75:
#                 tag_number += category_index_NE[pred_class]['name']
#                 match = 1
#             elif x1_cr-x1_ne > average_width//2:
#                 f_pred = sorted_ne[i][1]
#                 tag_number += category_index_NE[f_pred]['name']
#                 match = 1
#             else:
#                 pass
#             if match == 0:
#                 tag_number += "_"

#     ne_ends_at_coordinate = sorted_ne[-1][0][3]
#     cr_ends_at_coordinate = sorted_cr[-1][0][3]
#     if ne_ends_at_coordinate - cr_ends_at_coordinate > average_width/2:
#         last_pred = sorted_ne[-1][1]
#         tag_number += category_index_NE[last_pred]['name']

#     return tag_number


def get_matching_score(ground_truth_str, predicted_str, n_match):
    match_total = 0
    for i, char in enumerate(ground_truth_str):
        if i > len(predicted_str)-1:
            break
        if char == predicted_str[i]:
            match_total +=1
    print(match_total)
    if match_total >= n_match:
        return True
    else:
        return False


def nms(rects, thd=0.8, max_prob=0.0):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []
    remove = [False] * len(rects)
    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def remove_low_confidence_boxes(boxes, scores, classes, box_th=0.0):
    boxes_list = []
    scores_list = []
    classes_list = []

    for b,s,c in zip(boxes, scores, classes):
        if s >= box_th:
            boxes_list.append(b)
            scores_list.append(s)
            classes_list.append(c)

    return boxes_list, scores_list, classes_list


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


# # **Next function is the one that you can use to run inference and save results into a file:**

# # In[ ]:


def suppress_tag_label(detections, label_id_offset, category_index):
    # print(detections)
    # detections['detection_classes']:
    tag_found = False
    for k, v in category_index.items():
        # print(cate_id)
        # print(len(cate_id))
        # print(type(cate_id))
        if v['name'] == 'tag':
            tag_id = v['id']
            tag_found =  True
            break
    if not tag_found:
        return detections
    print(f"tag_id: {tag_id}")
    print(f"detection classes: {detections['detection_classes']}")
    # indices = [i for i, x in enumerate(detections['detection_classes']) if x == tag_id]
    indices = [i for i, x in enumerate(detections['detection_classes']) if x == tag_id-label_id_offset]
    # indices = [i for i, x in enumerate(detections) if x[2] == tag_id-label_id_offset]

    print(f"indices: {indices}")

    updated_detections = {"detection_classes": [], "detection_scores": [], "detection_boxes" : []}
    for i in range(len(detections['detection_classes'])):
        if i in indices:
            continue
        #print(detections['detection_classes'])
        #print(detections['detection_scores'])
        #detections['detection_scores'][i] = 0.0
        updated_detections['detection_classes'].append(detections['detection_classes'][i])
        updated_detections['detection_scores'].append(detections['detection_scores'][i])
        updated_detections['detection_boxes'].append(detections['detection_boxes'][i])

 
    updated_detections['detection_classes'] = np.array(updated_detections['detection_classes'])
    updated_detections['detection_scores'] = np.array(updated_detections['detection_scores'])
    updated_detections['detection_boxes']  = np.array(updated_detections['detection_boxes'])

    # print(updated_detections['detection_classes'])
    # print(updated_detections['detection_scores']) 
    # print(updated_detections['detection_boxes'])

    return updated_detections



def validate_output(annotation_info, detections, label_id_offset, box_th, stat, category_index=None, overlap_area_th=0.3):

    print(" Data Types :")
    print(type(detections['detection_boxes']))
    print(type(detections['detection_scores']))
    print(type(detections['detection_classes']))

    detections = suppress_tag_label(detections, label_id_offset, category_index)
    output_info = list(zip(detections['detection_boxes'],
                           detections['detection_scores'],
                           detections['detection_classes'] + label_id_offset
                          )
                      )

    boxes, scores, classes = nms(output_info, max_prob=0.0)
    # print(f"scores: {scores}")
    # print(f"classes: {classes}")
    boxes, scores, classes = remove_low_confidence_boxes(boxes, scores, classes, box_th)
    # print(f"scores: {scores}")
    # print(f"classes: {classes}")

#     #x1 = boxes[1]
#     #y1 = boxes[0]
#     #x2 = boxes[3]
#     #y2 = boxes[2]

#     #boxes[1], boxes[0], boxes[3], boxes[2] = boxes[0], boxes[1], boxes[2], boxes[3]
    detections['detection_boxes'] = np.array(boxes) # format: [y1, x1, y2, x2]
    detections['detection_scores'] = np.array(scores)
    detections['detection_classes'] = np.array(classes)
#     # print(detections['detection_boxes'])
#     # print(detections['detection_scores'])
    # print(len(detections['detection_classes']), detections['detection_classes'])
#     # print(annotation_info)

    correct_count = 0 # TP
    wrong_count = 0 # FP
    missed_count = 0 # FN

    total_characters = len(annotation_info)
    for annotation in annotation_info:
        bbox_gt = annotation[0] # bbox_gt format: (x1,y1,x2,y2)
#         #print(annotation[1])

        multiple_entry = 0
        if str(annotation[1]) in stat:
                stat[str(annotation[1])][0] += 1
        for bbox_pred, category in zip(boxes, classes):
#             # bbox_pred format: [y1, x1, y2, x2]
            x1 = bbox_pred[1]
            y1 = bbox_pred[0]
            x2 = bbox_pred[3]
            y2 = bbox_pred[2]

            bbox_pred = [x1,y1,x2,y2] # bbox_pred format: (x1,y1,x2,y2)
            overlap_area = intersection(bbox_gt, bbox_pred)/ min(square(bbox_gt), square(bbox_pred))
#             # print("Overlap: {}".format(overlap_area))
            if overlap_area > overlap_area_th:
#                 # print(f"category: {category}")
#                 # print(f"category: {type(category)}")
#                 # print(f"annotation[1]: {annotation[1]}")
#                 # print(f"annotation[1]: {type(annotation[1])}")
                multiple_entry += 1
                if (category == annotation[1]):
                    correct_count += 1
                    if str(annotation[1]) in stat:
                        stat[str(annotation[1])][1] += 1
                else:
                    wrong_count += 1
                    print(f"category: {category}")
                    print(f"(annotation[1]) : {(annotation[1])}")
                    if str(annotation[1]) in stat:
                        stat[str(annotation[1])][2] += 1

    stat['total'] += total_characters
    stat['total_correct'] += correct_count
    stat['total_wrong'] += wrong_count
    missed_count = total_characters - (correct_count + wrong_count)
    correct_percentage = (correct_count/total_characters)*100
    wrong_percentage = (wrong_count/total_characters)*100
    stat['correct_percent'].append(correct_percentage)
    stat['wrong_percent'].append(wrong_percentage)
    print("C: {}, W: {}, M: {}, multiple: {}, t: {}".format(correct_count, wrong_count, missed_count, multiple_entry, total_characters))
    return detections, stat