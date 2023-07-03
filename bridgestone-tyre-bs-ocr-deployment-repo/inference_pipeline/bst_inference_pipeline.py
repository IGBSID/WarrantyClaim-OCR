from cgitb import handler
import logging
from logging import handlers
from pickletools import uint8
# from metaflow import FlowSpec, step, IncludeFile, Parameter
# import efficientnet.tfkeras
import tensorflow as tf
import random
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image as Img
import cv2
import argparse
import os
from tqdm import tqdm
import numpy as np
import configparser
import glob
import sys
import csv
from albumentations import CLAHE
import torch
import torchvision
import time
import cv2
# import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence



log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Hello logging!")

category_index_CR = {0: {'id': 0, 'name': 'char'}, 1: {'id': 1, 'name': 'alp'}}

category_index_NE = {0: {'id': 0, 'name': '0'}, 1: {'id': 1, 'name': '1'},
                     2: {'id': 2, 'name': '2'}, 3: {'id': 3, 'name': '3'},
                     4: {'id': 4, 'name': '4'}, 5: {'id': 5, 'name': '5'},
                     6: {'id': 6, 'name': '6'}, 7: {'id': 7, 'name': '7'},
                     8: {'id': 8, 'name': '8'}, 9: {'id': 9, 'name': '9'},
                    10: {'id': 10, 'name': 'A'}, 11: {'id': 11, 'name': 'B'}, 
                    12: {'id': 12, 'name': 'C'}, 13: {'id': 13, 'name': 'D'},
                    14: {'id': 14, 'name': 'E'}, 15: {'id': 15, 'name': 'F'},
                    16: {'id': 16, 'name': 'G'}, 17: {'id': 17, 'name': 'H'}, 
                    18: {'id': 18, 'name': 'I'}, 19: {'id': 19, 'name': 'J'}, 
                    20: {'id': 20, 'name': 'K'}, 21: {'id': 21, 'name': 'L'}, 
                    22: {'id': 22, 'name': 'M'}, 23: {'id': 23, 'name': 'N'}, 
                    24: {'id': 24, 'name': 'O'}, 25: {'id': 25, 'name': 'P'}, 
                    26: {'id': 26, 'name': 'Q'}, 27: {'id': 27, 'name': 'R'}, 
                    28: {'id': 28, 'name': 'S'}, 29: {'id': 29, 'name': 'T'}, 
                    30: {'id': 30, 'name': 'U'}, 31: {'id': 31, 'name': 'V'}, 
                    32: {'id': 32, 'name': 'W'}, 33: {'id': 33, 'name': 'X'}, 
                    34: {'id': 34, 'name': 'Y'}, 35: {'id': 35, 'name': 'Z'}, 
                    36: {'id': 36, 'name': 'tag'}}


class Model():
    def __init__(self):
        self.NE_model_path = os.environ['NEModelPath']
        self.box_threshold = float(os.environ["boxTh"])
        self.CR2_model_path = os.environ["CR2ModelPath"]
        self.classifier_model_path = os.environ["ClassifierModelPath"]
        self.TF_path2scripts = os.environ["TFPath2Scripts"]
        sys.path.insert(0, self.TF_path2scripts)
        self.CR = self.load_CR_model()
        self.NE = self.load_NE_model()
        self.classifier = self.load_classifier_model()

     

    def get_CR_detections(self, input_image, box_th):
        image_with_detections, detections = self.object_detection(input_image, self.CR, box_th, mytask="CR")

        return input_image, image_with_detections, detections

    
    def get_NE_detections(self, input_image, box_th):
        image_with_detections, detections = self.object_detection(input_image, self.NE, box_th, mytask="NE")
        
        return input_image, image_with_detections, detections


    def get_classifier_output(self, cropped_box):
        cropped_box = cv2.resize(cropped_box, (172, 228))
        img_array = Img.img_to_array(cropped_box)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)
        pred = self.classifier.predict(img_preprocessed, verbose=0)

        name_list = ['0', '1', '2', "3", "4", "5", "6", "7", "8", "9", 'A', 'B', 'C', 'D', 'E', 
                    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', '*', 'O', 'P', 'Q',
                    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        index = pred.argmax()
        value = pred[0][index]
        return name_list[index], value


    def nms_pytorch(self, P ,thresh_iou=0.5):
        x1 = P[:, 0]
        y1 = P[:, 1]
        x2 = P[:, 2]
        y2 = P[:, 3]
        scores = P[:, 4]
        # calculate area of every block in P
        areas = (x2 - x1) * (y2 - y1)
        # sort the prediction boxes in P
        order = scores.argsort()
        # initialise an empty list for 
        keep = []
        
        while len(order) > 0:
            idx = order[-1]
            # push S in filtered predictions list
            keep.append(P[idx])
            # remove S from P
            order = order[:-1]
            # sanity check
            if len(order) == 0:
                break
        # select coordinates of BBoxes according to 
            xx1 = torch.index_select(x1,dim = 0, index = order)
            xx2 = torch.index_select(x2,dim = 0, index = order)
            yy1 = torch.index_select(y1,dim = 0, index = order)
            yy2 = torch.index_select(y2,dim = 0, index = order)
            
            # find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])
        
        # find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1
            
            # take max with 0.0 to avoid negative w and h
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            
            # find the intersection area
            inter = w*h
            
        # find the areas of BBoxes according the indices in order
            rem_areas = torch.index_select(areas, dim = 0, index = order) 
            
            # find the union of every prediction T in P
            # with the prediction S
            union = (rem_areas - inter) + areas[idx]
            
            # find the IoU of every prediction in P with S
            IoU = inter / union
            # print(IoU)
            
            # keep the boxes with IoU less than thresh_iou
            mask = IoU < thresh_iou
            order = order[mask]
        
        return keep


    def remove_low_confidence_boxes(self, boxes, scores, classes, box_th):
        boxes_list = []
        scores_list = []
        classes_list = []

        for b,s,c in zip(boxes, scores, classes):
            if s >= box_th:
                boxes_list.append(b)
                scores_list.append(s)
                classes_list.append(c)

        return boxes_list, scores_list, classes_list


    def object_detection(self, frame, model, box_th, mytask):
        if mytask == 'NE':
            names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
                    9: "9", 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G",
                    17: "H", 18: "I", 19: "J", 20: "K", 21: "L", 22: "M", 23: "N",
                    24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V",
                    32: "W", 33: "X", 34: "Y", 35: "Z"}
        else:
            names = {0: "C", 1: "A"}
        height, width, _ = frame.shape
        results = model(frame)
        boxes = results.pred[0][:,0:4]
        scores = results.pred[0][:,4]
        classes = results.pred[0][:,5]
        iou = 0.4
        det = self.nms_pytorch(results.pred[0], iou)
        boxes_list, scores_list, classes_list = self.remove_low_confidence_boxes(det, scores, classes, box_th)
        temp = 0
        for i, d in enumerate(boxes_list): #d = (x1, y1, x2, y2, conf, cls)
            if len(d):
                x1 = int(d[0].item())
                y1 = int(d[1].item())
                x2 = int(d[2].item())
                y2 = int(d[3].item())

                conf = boxes_list[-1]
                c = classes_list[i]
                detected_name = names[int(c)]

                if mytask == 'NE':
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    # frame = cv2.putText(frame, f'{detected_name}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2 , cv2.LINE_AA)
                else:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,100,0), 3)
        detections = boxes_list
        return frame, detections


    def load_CR_model(self):
        #TODO avoid hardcoding
        print(f"self.CR2_model_path: {self.CR2_model_path}")
        detection_model = torch.hub.load('/home/inference_pipeline/', 'custom', self.CR2_model_path, source='local')
        return detection_model


    def load_NE_model(self):
        #TODO avoid hardcoding
        detection_model = torch.hub.load('/home/inference_pipeline/', 'custom', self.NE_model_path, source='local')
        return detection_model


    def load_classifier_model(self):
        classifier_model = tf.keras.models.load_model(self.classifier_model_path)
        return classifier_model


class TagNumberExtractionPipeline():

    def __init__(self, models):
        self.models = models
       

    def visualise(self, image, mask_image, straight_contours):
        if mask_image is not None:
            if len(straight_contours):
                cv2.drawContours(image=image, contours=[np.array(straight_contours)], contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            out = np.concatenate((image, mask_image), axis=0)
            mask_image = out 
        return image, mask_image, straight_contours


    def get_CR_detections(self, image, viz_image, box_th):
        box_th = 0.7
        image, crop_image, detections = self.models.get_CR_detections(image, box_th)
        return image, detections


    def get_NE_detections(self, image, viz_image, box_th):
        image, crop_image, detections = self.models.get_NE_detections(image, box_th)

        return image, detections


    def get_classifier_output(self, cropped_box):
        detected_char = self.models.get_classifier_output(cropped_box)

        return detected_char


    def intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea


    def square(self, rect):
        """
        Calculates square of rectangle
        """
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


    def output_matching(self, detections_CR, detections_NE, image):
        tag_number = ""
        sorted_cr = sorted(detections_CR, key=lambda x: x[0])  #boxes sorted by x1
        try:
            sorted_cr = pad_sequence(sorted_cr, batch_first=True)
        except:
            pass
        sorted_ne = sorted(detections_NE, key=lambda x: x[0])
        sorted_ne_copy = sorted_ne.copy()
        try:
            sorted_ne = pad_sequence(sorted_ne, batch_first=True)
        except:
            pass
        
        width_list = []
        if sorted_cr == []:
            return "no_detection"

        width_list = [x2.item()-x1.item() for x1, y1, x2, y2 in sorted_cr[:,:4]]
        average_width = sum(width_list)/len(width_list)
        temp = []
        for i in range(len(sorted_cr)):
            bbox_CR_ = sorted_cr[i][:6]
            x1_cr = bbox_CR_[0]
            y1_cr = bbox_CR_[1]
            x2_cr = bbox_CR_[2]
            y2_cr = bbox_CR_[3]
            box_w = x2_cr - x1_cr
            box_h = y2_cr - y1_cr
            margin_w_cr = 0.09 * box_w
            margin_h_cr = 0.09 * box_h
            match = 0

            y1_cr_bound = int(y1_cr-margin_h_cr) if int(y1_cr-margin_h_cr) >= 0 else int(y1_cr)
            y2_cr_bound = int(y2_cr+margin_h_cr) if int(y2_cr+margin_h_cr) >= 0 else int(y2_cr)
            x1_cr_bound = int(x1_cr-margin_w_cr) if int(x1_cr-margin_w_cr) >= 0 else int(x1_cr)
            x2_cr_bound = int(x2_cr+margin_w_cr) if int(x2_cr+margin_w_cr) >= 0 else int(x2_cr)

            if temp:
                try:
                    for info in temp:
                        sorted_ne_copy.remove(info)
                except: pass
                temp = []
            for i in range(len(sorted_ne_copy)):
                bbox_NE_ = sorted_ne_copy[i][:4]
                pred_class = sorted_ne_copy[i][-1].item()
                NE_conf = sorted_ne_copy[i][-2].item()
                x1_ne = bbox_NE_[0]
                y1_ne = bbox_NE_[1]
                x2_ne = bbox_NE_[2]
                y2_ne = bbox_NE_[3]
                box_w = x2_ne - x1_ne
                box_h = y2_ne - y1_ne
                margin_w_ne = 0.09 * box_w
                margin_h_ne = 0.09 * box_h

                y1_ne_bound = int(y1_ne-margin_h_ne) if int(y1_ne-margin_h_ne) >= 0 else int(y1_ne)
                y2_ne_bound = int(y2_ne+margin_h_ne) if int(y2_ne+margin_h_ne) >= 0 else int(y2_ne)
                x1_ne_bound = int(x1_ne-margin_w_ne) if int(x1_ne-margin_w_ne) >= 0 else int(x1_ne)
                x2_ne_bound = int(x2_ne+margin_w_ne) if int(x2_ne+margin_w_ne) >= 0 else int(x2_ne)


                if x1_cr-x1_ne > average_width/2:
                    NE_pred_char = category_index_NE[pred_class]['name']   
                    cropped_box = image[y1_ne_bound:y2_ne_bound, x1_ne_bound:x2_ne_bound]
  
                    classifier_pred_char, classifier_pred_conf = self.get_classifier_output(cropped_box)

                    if classifier_pred_conf >= 0.95 and classifier_pred_char != '*':
                        tag_number += classifier_pred_char
                    elif classifier_pred_conf < 0.95 and NE_conf >= 0.8:
                        tag_number += NE_pred_char
                    else:
                        if classifier_pred_char != '*':
                            tag_number += classifier_pred_char

                    temp.append(sorted_ne_copy[i])
                    match = 1
                    continue
                overlap = self.intersection(bbox_CR_, bbox_NE_)/ min(self.square(bbox_CR_), self.square(bbox_NE_))
                if overlap >= 0.75:
                    NE_pred_char = category_index_NE[pred_class]['name']  
                    
                    cropped_box = image[y1_cr_bound:y2_cr_bound, x1_cr_bound:x2_cr_bound]
                     
                    classifier_pred_char, classifier_pred_conf = self.get_classifier_output(cropped_box)

                    if classifier_pred_conf >= 0.95 and classifier_pred_char != '*':
                        tag_number += classifier_pred_char
                    elif classifier_pred_conf < 0.95 and NE_conf >= 0.8:
                        tag_number += NE_pred_char
                    else:
                        if classifier_pred_char != '*':
                            tag_number += classifier_pred_char

                    temp.append(sorted_ne_copy[i])
                    match = 1
                    break
                else:
                    match = 0
            if match == 0:
                cropped_box = image[y1_cr_bound:y2_cr_bound, x1_cr_bound:x2_cr_bound]
                              
                classifier_pred_char, classifier_pred_conf = self.get_classifier_output(cropped_box)

                if classifier_pred_char != '*':
                    tag_number += classifier_pred_char

        return tag_number


    def run(self, image_file, box_th):
        image = cv2.imread(image_file)
        viz_image = image.copy()
        img_another_copy = image.copy()
        viz_image, detections_CR = self.get_CR_detections(image, viz_image, box_th)

        viz_image, detections_NE = self.get_NE_detections(image, viz_image, box_th)

        tag_n = self.output_matching(detections_CR, detections_NE, img_another_copy)
    
        return viz_image , tag_n
        

if __name__ == "__main__":
    m = Model()
    t = TagNumberExtractionPipeline(m)
    box_th = 0.8

    ocr_output = []
    for image_file in os.listdir("/home/images/"):
        print(image_file)
        image_file = os.path.join("/home/images", image_file)
        image_name = image_file.rpartition("/")[2]

        image_file_split = image_file.split(".j")[0]
        another_split_list = image_file_split.split('_')
        tag_number_gt = another_split_list[-2]
        mask_image, tag_number = t.run(image_file, box_th)
        print(tag_number)
        cv2.imwrite(os.path.join("/home/output/", image_name), mask_image)
        ocr_output.append({"FileName": tag_number_gt, "OCR_prediction": tag_number})
        
    fieldnames = ['FileName', 'OCR_prediction']

    with open('/home/output/model_output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ocr_output)
