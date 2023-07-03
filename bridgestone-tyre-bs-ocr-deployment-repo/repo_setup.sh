#! /bin/bash
export NEModelPath=/home/TF_docker_setup/Input/Model/NE/_best_NE.pt
export NELabelmapPath=/home/Data/label_map_NE.pbtxt
CR2ModelPath=/home/TF_docker_setup/Input/Model/CR/_best_CR.pt
CR2LabelmapPath = /home/Data/label_map_CR2.pbtxt
TFPath2Scripts = /home/Tensorflow/models/research/
boxTh = 0.8
ClassifierModelPath = /home/TF_docker_setup/Input/Model/Classifier/eff_b0_v11_37_rect2.h5
