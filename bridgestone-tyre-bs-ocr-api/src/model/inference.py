# import json
# from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
# from msrest.authentication import ApiKeyCredentials
# import os
# from werkzeug.utils import secure_filename
# import requests

# def get_tsn_ocr(filepath):
#     ocr_endpoint = os.environ['OCR_ENDPOINT']
#     prediction_key = os.environ['PREDICTION_KEY']
#     project_id = os.environ['PROJECT_ID']
#     publish_iteration_name = os.environ['PUBLISH_ITERATION_NAME']

#     prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
#     # predictor = CustomVisionPredictionClient(damage_category_endpoint, prediction_credentials)

#     with open(os.path.join (filepath), "rb") as image_contents:
#         print("=================",image_contents.name)
#         filename = secure_filename(image_contents.name)
#         # file_names.append(filename)
#         # input_data = {"tyre_number": ""}
#         print(filename)
#         resp = requests.post(ocr_endpoint,files={'file':(image_contents.name, image_contents.stream, image_contents.content_type, image_contents.headers)})
#         json_resp = json.loads(resp.text)
#         return json_resp