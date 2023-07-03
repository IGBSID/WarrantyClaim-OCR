from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
import argparse
import csv
import textdistance
import json
import editdistance
import configparser

def get_damage_labels(data_dir):
    print(os.listdir(data_dir))
    return os.listdir(data_dir)

def get_best_matching_dirname(key, dirs):
    token_1 = key.lower().split()
    max_score = 0.0
    max_score_token = None

    for d in dirs:
        token_2 = d.lower().split()
        match_score = textdistance.ratcliff_obershelp(token_1 , token_2)
        if match_score > max_score:
            max_score = match_score
            max_score_token = d
    return max_score_token

def get_best_matching_dirname_editdistance(key, dirs):
    token_1 = key.lower()
    min_score = 100
    min_score_token = None

    for d in dirs:
        token_2 = d.lower()
        match_score = editdistance.eval(token_1 , token_2)
        #print(match_score, token_1, token_2)
        if match_score < min_score:
            min_score = match_score
            min_score_token = d
    print("----")
    return min_score_token

def get_best_matching_dirname_directmatch(key, dirs):
    token = None
    token_1 = key.lower()
    for d in dirs:
        token_2 = d.lower()
        #print(token_1, "--", token_2)
        if token_2 == token_1:
            token = d
    return token

def labels_to_values(noisylabels, labelmap):
    values = []

    for l in noisylabels:
        min_score_token_value = None
        for k,v in labelmap.items():
            if l == k:
                min_score_token_value = v

        if min_score_token_value is not None:
            values.append(str(min_score_token_value))

    return values

def update_label_map(damage_dir_names, labelmap):
    new_labelmap = {}
    for key,v in labelmap.items():
        #new_key = get_best_matching_dirname(key, damage_dir_names)
        new_key = get_best_matching_dirname_directmatch(key, damage_dir_names)
        print("update: ", new_key, v)
        if new_key is not None:
            if new_key not in new_labelmap:
                new_labelmap[new_key] = v
    return new_labelmap

def Print(message, debug=False):
    if debug:
        print(message)

class CustomVisionClassifier():
    def __init__(self, name=""):
        ENDPOINT = os.environ['DAMAGE_CATEGORY_ENDPOINT']
        prediction_key = os.environ['PREDICTION_KEY']
        self.project_id = os.environ["PROJECT_ID"+"_"+name]
        self.publish_iteration_name = os.environ["PUBLISH_ITERATION_NAME"+"_"+name]

        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    def predict(self, img_path, top1=True):
        '''
        final_results will be list-of-list
        [ [class1],[probability],
          [class2],[probability],
          ...
        ]
        '''
        with open(img_path, "rb") as image_contents:
            results = self.predictor.classify_image(
                            self.project_id,
                            self.publish_iteration_name,
                            image_contents.read()
                      )

            final_results = []
            for prediction in results.predictions:
                final_results.append( [prediction.tag_name, prediction.probability * 100])

            if top1:
                return final_results[0]
            return final_results

class CascadeClassifier():
    def __init__(self):
        self.c0 = CustomVisionClassifier("C0")
        self.c1 = CustomVisionClassifier("C1")
        self.c2 = CustomVisionClassifier("C2")
        self.c3 = CustomVisionClassifier("C3")
        self.c4 = CustomVisionClassifier("C4")
        self.c5 = CustomVisionClassifier("C5")

    def predict(self, img_path, debug=False):
        response = list()
        # C0: Check if input image is of Tyre
        r0 = self.c0.predict(img_path)
        Print("C0:{}".format(r0[0]), debug)
        if r0[0] != "Yes":
            return response
        # C1: 7 Damages vs Others
        r1 = self.c1.predict(img_path, top1=False)
        Print("  C1:{}".format(r1[0][0]), debug)
        if r1[0][0] != "Others":
            response.append(r1[0])
            return response

        # C2: Sidewall damages vs Tread Damages vs Other Seperations
        r2 = self.c2.predict(img_path, top1=False)
        Print("    C2:{}".format(r2[0][0]), debug)
        # C3: Sidewall damages
        if r2[0][0] == "SideWallDamage":
            r3 = self.c3.predict(img_path, top1=False)
            response.append(["SideWallDamage",0.0])
            for r in r3:
                Print("     C3:{}".format(r), debug)
                response.append(r)
            return response

        # C4: Sidewall damages vs Tread Damages vs Other Seperations
        elif r2[0][0] == "TreadDamage":
            r4 = self.c4.predict(img_path, top1=False)
            response.append(["TreadDamage", 0.0])
            for r in r4:
                Print("     C4:{}".format(r), debug)
                response.append(r)
            return response

        # C5: Sidewall damages vs Tread Damages vs Other Seperations
        else:
            r5 = self.c5.predict(img_path, top1=False)
            response.append(["OtherSeparations", 0.0])
            for r in r5:
                Print("     C5:{}".format(r), debug)
                response.append(r)
            return response
        return list()

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--testdata", help = "Path to test data dir",required=True)
    parser.add_argument("-l", "--labelmap", help = "Path to labelmap",required=True)
    #parser.add_argument("-n", "--namemap", help = "Path to labelmap")
    parser.add_argument("-c", "--configFile", help = "Path to config file",required=True)
    args = parser.parse_args()

    if not os.path.exists(args.configFile):
        assert False, f"Config File Not Found {args.configFile}"

    print(args.configFile)
    print(type(args.configFile))
    CC = CascadeClassifier(args.configFile)

    damage_test_labels = get_damage_labels(args.testdata)
    print(damage_test_labels)
    print("Found {} directories".format(len(damage_test_labels)))

    labelmap = None
    with open(args.labelmap) as json_file:
        labelmap = json.load(json_file)

    new_labelmap = update_label_map(damage_test_labels, labelmap)

    for k,v in new_labelmap.items():
        print(k,v)

    assert len(damage_test_labels) == len(new_labelmap)

    header = ["filename", "1"]
    with open('damage_classification_evaluation.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for l in damage_test_labels:
            dir_path = os.path.join(args.testdata,l)

            for img in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img)
                print(img_path)
                row= []
                row.append(l)

                results = CC.predict(img_path, debug=True)

                # Display the results.
                if len(results):
                    row.append(results[0][0])
                else:
                    print("Non Tyre Image")
                    continue

                newrow = labels_to_values(row, new_labelmap)
                print(row)
                print(len(newrow), len(row))
                print(newrow)
                assert len(newrow) == len(row)
                print(newrow)
                writer.writerow(newrow)
                time.sleep(0.23)