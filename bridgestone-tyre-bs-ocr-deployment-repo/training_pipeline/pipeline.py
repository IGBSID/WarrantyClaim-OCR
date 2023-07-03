from metaflow import FlowSpec, step, IncludeFile, Parameter
import os
from zipfile import ZipFile
import shutil
import re
import glob
import json
import tarfile
import app.utils.helper_aws as aws
import zipfile
#import model_validation

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


class TrainingPipelineFlow(FlowSpec):
    """
    """

    downloadModelDir = Parameter(
        "downloadModelDir",
        help="",
        required=True
    )

    dataDownLoadDir = Parameter(
        "dataDownLoadDir",
        help="",
        required=True
    )

    configFile = Parameter(
        "configFile",
        help="",
        required=True
    )

    numsteps = Parameter(
        "numsteps",
        help="",
        required=True
    )

    dryrun = Parameter(
        "dryrun",
        type = bool,
        help="",
        default=True
    )

    def download_efficient_net_b4(self, flag):
        make_dir(self.downloadModelDir)

        filename = os.path.join(self.downloadModelDir, "{}.tar.gz".format(self.my_model))
        if not os.path.exists(filename):
            download_model = "wget http://download.tensorflow.org/models/object_detection/tf2/20200711/{}.tar.gz \
            --directory-prefix={}".format(self.my_model, self.downloadModelDir)
            os.system(download_model)

            with tarfile.open(os.path.join(self.downloadModelDir,"{}.tar.gz".format(self.my_model))) as f:
                f.extractall(self.downloadModelDir)



    @step
    def start(self):
        """
        """
        print("Starting Trainiing !!!")

        self.my_model      = "efficientdet_d4_coco17_tpu-32"
        self.tf_train_data = "tf_train_aug_sep20" #edit this
        self.tf_test_data  = "tf_test" #edit this
        self.test_images   = "test_set_aug_sep20" #edit this
        self.files_to_download = [self.tf_train_data, self.tf_test_data, self.test_images]
        self.download_efficient_net_b4(flag=True)
        self.configFileDir = "/home/experiments"
        self.labelmap_NE      = "/home/tyre-inspection/Data/label_map_NE.pbtxt"
        self.labelmap_CR      = "/home/tyre-inspection/Data/label_map_CR.pbtxt"
        self.labelmap_NE_CR      = "/home/tyre-inspection/Data/label_map_NE_CR.pbtxt"
        self.next(self.download_s3_data)

    @step
    def download_s3_data(self):
        for f in self.files_to_download:
            download_dir = os.path.join(self.dataDownLoadDir, f)
            make_dir(download_dir)
            file_to_download = os.path.join(download_dir, "{}.zip".format(f))
            file_on_s3  = "agumentations/{}.zip".format(f) #edit this

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

    @step
    def wait_to_upload_pipeline_config(self):
        #make_dir(self.configFileDir)
        #message ="Please keep config file(s) under dir {}".format(self.configFileDir)
        #t=input(message)
        self.next(self.start_training)

    @step
    def start_training(self):
        print(self.configFile)

        self.modelsavedir = os.path.join(self.downloadModelDir, "trained_model")
        trainingscriptlocation = "/home/Tensorflow/models/research/object_detection/model_main_tf2.py"
        make_dir(self.modelsavedir)

        train_model = "python {} --pipeline_config_path={} --num_train_steps={} --model_dir={} \
                --alsologtostder".format(trainingscriptlocation, self.configFile , \
                 self.numsteps, self.modelsavedir)
        os.system(train_model)
        self.next(self.export_trained_model)


    @step
    def export_trained_model(self):

        exportscriptlocation = "/home/Tensorflow/models/research/object_detection/exporter_main_v2.py"
        trainedcheckpointdir = self.modelsavedir

        self.exportdir = os.path.join(self.downloadModelDir, "exported_model")
        make_dir(self.exportdir)

        export_model = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} \
         --output_directory={} --input_type=image_tensor".format(exportscriptlocation, self.configFile,\
             trainedcheckpointdir, self.exportdir)

        os.system(export_model)
        self.next(self.end)

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