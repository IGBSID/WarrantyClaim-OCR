import json
from src.upload.upload_to_cloud import AzureCloud
from database.models.api_data import insert_or_update_api_request_data, update_api_response_data, update_validate_request
from database.models.image_data import insert_image_data, update_image_data, delete_image_data
from src.helpers.error import UploadToAzureException, errors
import os

def upload_to_azure(filepath, azure_folder_name):
    try:
        azure_connection_string = os.environ['AZURE_CONNECTION_STRING']

        azure_container_name = os.environ['AZURE_CONTAINER_NAME']

        image_url, status = AzureCloud.upload(azure_connection_string, azure_container_name, filepath, azure_folder_name, None)
        return image_url, status
    except (UploadToAzureException, Exception) as ex:
        print("error msg = " + str(ex))
        print("error during upload to Azure")
        raise ex

def insert_or_update_request_in_db(request_data, api_endpoint):
    try:
        request_data["api_endpoint"] = api_endpoint
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
        print("api_response_data not inserted in db")    
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

def create_response(code, resp):
    try:
        return_message = errors[code]
        return_message["ocr_result"] = resp
        return json.dumps(return_message)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in create_response")  
        raise ex

def update_validate_request_in_db(image_data, api_endpoint):
    try:
        return update_validate_request(image_data, api_endpoint)        
    except Exception as ex:
        print("error msg = " + str(ex))
        print("error in update_validate_request_in_db")  
        raise ex

def delete_image_in_db(image_data):
    try:
        return delete_image_data(image_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("image_data not deleted in db")  
        raise ex
