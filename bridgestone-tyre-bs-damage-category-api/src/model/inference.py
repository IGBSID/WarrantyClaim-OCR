import json
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from src.model.cascade_classifier import CascadeClassifier
import random

CC = CascadeClassifier()

def get_damage_labels():
    with open("resources/damage_label.json","r") as file:
        jsonData = json.load(file)
        resultkeys = list(jsonData.keys() )       
        resultValues = list(jsonData.values())
        return  resultkeys,resultValues

def get_damage_category(filepath):
    '''
    Results from CascadeClassifier will fall in either of following cases:

    Case 1: Empty list
     results = [] # Input image is non Tyre, unknow Error

    Case 2: Only One element in the list
      results = [[Class, probability score]], example [['BeadBurst', 98.00]]
      onle one element -> unmerged category
      Damage_name = results[0][0]

    Case3:  Multiple elements
      results = [ [class1, prob1] , [class2, prob2] ,  ... [] ]
      len(results)
        category_name = results[0][0]
        Damage_name = results[i][0]
        Damage_probabily = results[i][1]
    '''
    damage_category_endpoint = os.environ['DAMAGE_CATEGORY_ENDPOINT']
    prediction_key = os.environ['PREDICTION_KEY']
    project_id = os.environ['PROJECT_ID']
    publish_iteration_name = os.environ['PUBLISH_ITERATION_NAME']

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(damage_category_endpoint, prediction_credentials)

    results = CC.predict(filepath, debug='True')

    print(results)
    output_list = []
    #priority_levels = ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]
    # Case 2
    if len(results)==1:
        damage_name = results[0][0] #prediction.tag_name
        damage_priority = ""
        output_list.append([damage_name, damage_priority])
        
    # Case 3
    elif len(results)==3:
        index = random.choice([1,2])
        damage_name = results[0][0] #prediction.tag_name
        damage_probability = results[0][1]
        output_list.append([damage_name, damage_probability])

        for i in [1,2]:
            damage_name = results[i][0]
            damage_probability = results[i][1]
            output_list.append([damage_name, damage_probability])
    # Case 4
    elif len(results)==5:
        damage_name = results[0][0]
        damage_probability = results[0][1]
        output_list.append([damage_name, damage_probability])

        for i in [1,2,3,4]:
            damage_name = results[i][0]
            damage_probability = results[i][1]
            
            output_list.append([damage_name, damage_probability])

    
    #return output_list

    output_list1 = []
    for i in results:
        output2 = i[0] #prediction.tag_name
        output_list1.append(output2)
    print(output_list1,output_list)
    return output_list1,output_list,results