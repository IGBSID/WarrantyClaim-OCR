from metaflow import FlowSpec, step, IncludeFile, Parameter
import os
from zipfile import ZipFile
import shutil
import re
import glob
import json
import tarfile
import data_augmentation.app.utils.helper_aws as aws
import zipfile
#import model_validation
import configparser
import pathlib
from ast import literal_eval as make_tuple
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class LoadData(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img/255.0
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def jpeg_img_files(Dir):
    """Recursively iterate all the .jpeg files in the root directory and below"""
    imgpaths = []
    for path, dirs, files in os.walk(Dir):
        imgpaths += [os.path.join(path, file) for file in files if (pathlib.Path(file).suffix == '.jpeg' or pathlib.Path(file).suffix == '.jpg')]

    return imgpaths

def get_correspoding_image_and_mask_paths(imageDir, maskDir):

    pathMapping = list()

    imageNameList = jpeg_img_files(imageDir)
    print(len(imageNameList))

    maskNameList = jpeg_img_files(maskDir)
    print(len(maskNameList))

    for i,m in zip(imageNameList,maskNameList):
        #print(i,m)
        text_img = ''.join(i.split('/')[-4:])
        text_mask= ''.join(m.split('/')[-4:])
        if text_img != text_mask:
            assert False
        pathMapping.append((i,m))

    return imageNameList, maskNameList

class TrainingPipelineFlow(FlowSpec):
    """
    """

    trainImgDir = Parameter(
        "trainImgDir",
        help="",
        required=True
    )

    trainMaskDir = Parameter(
        "trainMaskDir",
        help="",
        required=True
    )

    valImgDir = Parameter(
        "valImgDir",
        help="",
        required=True
    )

    valMaskDir = Parameter(
        "valMaskDir",
        help="",
        required=True
    )

    configFile = Parameter(
        "configFile",
        help="",
        required=True
    )

    modelSaveDir = Parameter(
        "modelSaveDir",
        help="",
        required=True
    )


    @step
    def start(self):
        """
        """
        print("Starting Trainiing !!!")
        if not os.path.exists(self.configFile):
            assert False, f"Config File Not Found {self.configFile}"

        config = configparser.ConfigParser()
        config.read(self.configFile)

        self.imageSize = make_tuple(config["settings"]["Size"])
        self.batchSize = int(config["settings"]["BatchSize"])
        self.modelBackbone = config["settings"]["Backbone"]
        self.warmUpEpochs  = int(config["settings"]["WarmUpEpochs"])
        self.Epochs  = int(config["settings"]["Epochs"])
        print(f"Image size: {self.imageSize}")
        self.next(self.prepare_train_test_dataset_mappings)


    @step
    def prepare_train_test_dataset_mappings(self):
        """
        """
        trainImgDir = self.trainImgDir
        trainMaskDir = self.trainMaskDir

        valImgDir = self.valImgDir
        valMaskDir = self.valMaskDir

        if not os.path.exists(trainImgDir) and not os.path.exists(trainMaskDir):
            assert False, f"Path does not exist"

        trainImagePaths, trainMaskPaths = \
        get_correspoding_image_and_mask_paths(imageDir=trainImgDir, maskDir=trainMaskDir)

        self.trainDataLoader = LoadData(self.batchSize, self.imageSize,
                                        trainImagePaths, trainMaskPaths)

        self.trainDataSize = len(trainImagePaths)

        valImagePaths, valMaskPaths = \
        get_correspoding_image_and_mask_paths(imageDir=valImgDir, maskDir=valMaskDir)

        self.valDataLoader = LoadData(self.batchSize, self.imageSize,
                                        valImagePaths, valMaskPaths)

        self.next(self.train_specified_model_as_per_config)

    @step
    def train_specified_model_as_per_config(self):
        import segmentation_models as sm
        from segmentation_models.utils import set_trainable

        model = sm.Unet(self.modelBackbone,
                        input_shape = self.imageSize + (3,),
                        classes = 1,
                        encoder_weights='imagenet',
                        encoder_freeze=True,
                        activation='sigmoid')
        model.compile('Adam', sm.losses.bce_jaccard_loss, [sm.metrics.iou_score])

        model.fit(self.trainDataLoader,
            batch_size=self.batchSize,
            epochs=self.warmUpEpochs,
            validation_data=self.valDataLoader,
            )

        #set_trainable(model) # set all layers trainable and recompile model
        for layer in model.layers:
            layer.trainable = True

        #csv_logger = CSVLogger('log.csv', append=True, separator=';')
        modelName = "tag_{}_{}_{}_{}_".format(
             self.modelBackbone, self.Epochs, self.batchSize, self.trainDataSize
            )

        modelSaveName = modelName+"{epoch:02d}-{val_loss:.2f}.hdf5"
        modelSavePath = os.path.join(self.modelSaveDir, modelSaveName)
        saveModelCallBack = keras.callbacks.ModelCheckpoint(filepath=modelSavePath,
                                        save_best_only=True,
                                        monitor='val_loss')

        logSaveName = os.path.join(self.modelSaveDir, modelName+'log.csv')
        logsCallBack = keras.callbacks.CSVLogger(logSaveName, append=True, separator=';')

        callbacks = [ saveModelCallBack , logsCallBack ]

        model.fit(self.trainDataLoader,
            batch_size=self.batchSize,
            epochs= self.Epochs - self.warmUpEpochs,
            validation_data=self.valDataLoader,
            callbacks=callbacks
            )

        # tag_{backbone}_{unet}_{Epochs}_{warmUpepchs}_{batchSize}_{data_size}

        self.next(self.end)
    '''
    @step
    def download_s3_data(self):
        for f in self.files_to_download:
            download_dir = os.path.join(self.dataDownLoadDir, f)
            make_dir(download_dir)
            file_to_download = os.path.join(download_dir, "{}.zip".format(f))
            file_on_s3  = "agumentations/{}.zip".format(f)

            if not os.path.exists(file_to_download):
                aws.download_file(file_to_download, file_on_s3)

                with zipfile.ZipFile(file_to_download, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)


        self.tf_train_data_dir = os.path.join(self.dataDownLoadDir, self.tf_train_data + "/tf_train")
        self.tf_test_data_dir  = os.path.join(self.dataDownLoadDir, self.tf_test_data + "/tf_test")

        self.test_image_dir = os.path.join(self.dataDownLoadDir, self.test_images + "/test_set_extract")
        self.test_image_annotation_file = os.path.join(self.test_image_dir, "merged.json")

        message = "Paths: \n \
        train_tf_record: {} \n \
        test_tf_record: {}  \n \
        model_checkpoint: {} \n \
        label_map_NE: {} \n \
        label_map_CR: {} \n \
        ".format(self.tf_train_data_dir + "/coco_train.record-?????-of-00100",
                 self.tf_test_data_dir + "/coco_testdev.record-?????-of-00050",
                self.downloadModelDir +"/" +self.my_model+ "/checkpoint/ckpt-0",
                self.labelmap_NE,
                self.labelmap_CR)

        print(message)
        if self.dryrun:
            assert False

        self.next(self.wait_to_upload_pipeline_config)
    '''

    '''
    @step
    def model_validation_and_stats(self):

        modelcheckpointdir = os.path.join(self.exportdir, "checkpoint")
        configFile = os.path.join(self.exportdir,"pipeline.config")

        labelmap = self.labelmap_NE
        valimagespath = self.test_image_dir
        valjsonpath = self.test_image_annotation_file

        #model_validation.run_model(configFile, modelcheckpointdir, labelmap, valimagespath, valjsonpath)

        validation_script = "/home/tyre-inspection/training_pipeline/model_validation.py"

        validate_model = "python {} -c {} -m {} -l {} -i {} -a {} -t {}".format\
            (validation_script, configFile, modelcheckpointdir, labelmap, valimagespath,\
                valjsonpath, self.taskName)

        os.system(validate_model)

        self.next(self.end)
    '''
    @step
    def end(self):
        """
        """ 


if __name__ == "__main__":
    TrainingPipelineFlow()