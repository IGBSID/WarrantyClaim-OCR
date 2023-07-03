import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
path2outputdir = os.environ["Path2outputDir"]
path2images= os.environ["Path2images"]

from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import json
import base64
import cv2
import numpy as np

from inference_pipeline.bst_inference_pipeline import Model, TagNumberExtractionPipeline

m = Model()
print("All Models Loaded ...")
t = TagNumberExtractionPipeline(m)
print("Pipeline Created ...")

app = Flask(__name__)


def run_inference_pipeline(Imagefile):
    try:
        tag_number = ""
        mask_image = np.zeros((64,64,3), np.uint8)
        match_decision = None
        Decision = ""
        outimgfile = os.path.join(path2outputdir, "out.jpg")
        try:
            mask_image, tag_number = t.run(Imagefile, box_th=0.8)
        except Exception as ex:
            print("error msg = " + str(ex))
            print("Error in t.run")

        cv2.imwrite(outimgfile, mask_image)

    except Exception as ex:
        print("error msg = " + str(ex))
        print("Error in run_inference_pipeline") 

    return outimgfile, tag_number


def get_response(filename):
    try:
        resp = "empty response"
        outimgpath, tag_number = run_inference_pipeline(filename)
        resp = create_response(outimgpath, tag_number)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("Error in get_response") 
    return resp


@app.route('/tsn', methods = ['POST'])
def ocr_tsn():
    try:
        resp = "empty response"
        if request.method == 'POST':
            f = request.files['file']
            filename = os.path.join(path2images,secure_filename(f.filename))
            f.save(filename)

            resp = get_response(filename)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("Error in ocr_tsn")
    return resp


@app.route('/trn', methods = ['POST'])
def ocr_trn():
    try:
        resp = "empty response"
        if request.method == 'POST':
            f = request.files['file']
            filename = os.path.join(path2images,secure_filename(f.filename))
            f.save(filename)

            resp = get_response(filename)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("Error in ocr_trn")
    return resp


def create_response(f, tag_number):
    try:
        print("filename: ", f)
        with open(f, mode='rb') as file:
            img = file.read()

        a = {'extracted_value': tag_number, "out_file" : base64.b64encode(img).decode()}
    except Exception as ex:
        print("error msg = " + str(ex))
        print("Error in create_response")
    return json.dumps(a)


if __name__ == "__main__":
    app.run(debug=True)
