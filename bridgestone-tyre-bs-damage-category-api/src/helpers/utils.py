import json
from src.upload.upload_to_cloud import AzureCloud
from database.models.api_data import insert_or_update_api_request_data, update_api_response_data, update_validate_request, update_user_response_data
from database.models.image_data import insert_image_data, update_image_data, delete_image_data
from database.models.damage_category_master import get_category_code_from_name
from src.helpers.error import UploadToAzureException, errors
import os

def max_category(f_list):    
    try:
        return max(set(f_list), key = f_list.count)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in max_category")    
        raise ex


def upload_to_azure(filepath):
    try:
        azure_connection_string = os.environ['AZURE_CONNECTION_STRING']

        azure_container_name = os.environ['AZURE_CONTAINER_NAME']

        azure_folder_name = os.environ['AZURE_FOLDER_NAME']

        image_url, status = AzureCloud.upload(azure_connection_string, azure_container_name, filepath, azure_folder_name, None)
        return image_url, status
    except (UploadToAzureException, Exception) as ex:
        print("error msg = " + str(ex))
        print("error during upload to Azure")
        raise ex

def insert_or_update_request_in_db(request_data):
    try:
        request_data["api_endpoint"] = "damage_category"
        return insert_or_update_api_request_data(request_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("api_request_data not inserted in db")    
        raise ex

def update_response_in_db(response_data):
    try:
        return update_api_response_data(response_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("api_response_data not updated in db")    
        raise ex

def update_user_response_in_db(response_data):
    try:
        return update_user_response_data(response_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("user_response not updated in db")    
        raise ex


def insert_image_in_db(image_data):
    try:
        return insert_image_data(image_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("image_data not inserted in db")  
        raise ex

def update_image_in_db(image_data):
    try:
        return update_image_data(image_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("image_data not updated in db")  
        raise ex

def delete_image_in_db(image_data):
    try:
        return delete_image_data(image_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("image_data not deleted in db")  
        raise ex

def get_category_code_from_db(name):
    try:
        return get_category_code_from_name(name)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in category code")
        raise ex

def create_response(code, response={}):
    try:
        return_message = errors[code]
        if len(response.keys()) > 0:
            return_message["damage_category_code"] = response["damage_category_code"]  if response["damage_category_code"] is not None else ""
            return_message["damage_category_name"] = response["damage_category_name"]  if response["damage_category_name"] is not None else ""
            return_message["other_damage_category"] = response["other_damage_category"]  if response["other_damage_category"] is not None else ""
        else:
            return_message["damage_category_code"] = ""
            return_message["damage_category_name"] = ""
            return_message["other_damage_category"] = ""
        return json.dumps(return_message)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in create_response")  
        raise ex

def update_validate_request_in_db(image_data):
    try:
        return update_validate_request(image_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("validate request not updated in db")  
        raise ex

def get_cascade_damage_labels():
    with open("resources/cascade_damage_label.json","r") as file:
        jsonData = json.load(file)
        resultkeys = list(jsonData.keys() )       
        resultValues = list(jsonData.values())
        return  resultkeys,resultValues

def convert_sub_damage_category(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct