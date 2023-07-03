from src.helpers.error import TokenError, APIRequestIncorrectDataError
from src.helpers.auth import authenticate
from src.helpers.utils import create_response, insert_or_update_request_in_db, insert_image_in_db, upload_to_azure, update_image_in_db, update_response_in_db, delete_image_in_db
import os
import requests
import json

def get_ocr_trn(request):
    try:
        # get request data
        request_data = {}
        files = request.files.getlist('Image_List')
        request_data["no_of_images"] = request.form.get('No_Of_Images')
        request_data["record_id"] = request.form.get('Record_Id')
        request_data["sent_on"] = request.form.get('Sent_On')
        request_data["image_details"] = request.form.get('Image_Details')
        print("request.files=" + str(request.files))
        print("request.form=" + str(request.form))
        print("request_data=" + str(request_data))
        print("files=" + str(files))
                
        # validate the mandatory input in request data
        file_count = 0
        file_present = False
        for file in files:
            if not file:
                file_present = False
            else:
                file_present = True
            file_count += 1
        print("file_count=" + str(file_count))
        print("file_present=" + str(file_present))
        if  (request is None or  # no request
            not file_present or  # no file is passed 
            file_count != 1 or # more than 1 file is passed  
            request_data["no_of_images"] is None or \
            request_data["record_id"] is None or request_data["sent_on"] is None or \
            request_data["image_details"] is None):
            print("some error")
            raise APIRequestIncorrectDataError
    except (APIRequestIncorrectDataError, Exception) as ex:
        print("error msg = " + str(ex))
        print("API request data not correct")
        return create_response("APIRequestIncorrectDataError", "")    

    # authenticate header
    try:
        if (request.headers["Http-X-Api-Secret"] is not None):
            token = request.headers["Http-X-Api-Secret"]
            if (not authenticate(token)):
                raise TokenError
        else:
            raise TokenError
    except (TokenError, Exception) as ex:
        print(ex)
        return create_response("TokenError", "")           

    # insert request into db
    try:
        request_id, is_insert = insert_or_update_request_in_db(request_data, "trn")
        print("request_id = " + str(request_id))
    except Exception as ex:
        print("error msg = " + str(ex))
        print("API request data not added in database")
        return create_response("DatabaseInsertError", "")   

    if (not is_insert): #i.e. update, then delete all files
        try:
            image_data = {}
            image_data["api_request_id"] = request_id
            delete_image_in_db(image_data)
        except Exception as ex:
            print("error msg = " + str(ex))
            print("image data not deleted from database")
            return create_response("DatabaseDeleteError", "")     

    for file in files:
        if file:
            try:
                trn_ocr_endpoint = os.environ['TRN_OCR_ENDPOINT']

                # get damage results from the model            
                resp = requests.post(trn_ocr_endpoint, files={'file':(file.filename, file.stream, file.content_type, file.headers)})
                json_resp = json.loads(resp.text)
                resp = json_resp["extracted_value"]
            except Exception as ex:
                print("error msg = " + str(ex))
                print("Unable to receive OCR model response")
                return create_response("OCRModelError", "")           

            # insert image details into db
            try:
                image_data = {}
                image_data["api_request_id"] = request_id
                image_data["image_details"] = request_data["image_details"]
                image_data["created_on"] = request_data["sent_on"]
                image_id = insert_image_in_db(image_data)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("image data not added in database") 
                return create_response("DatabaseInsertError", "")     

            try:
                file_extension = file.filename.split(".")[1]
                file_name = str(image_id)+"."+str(file_extension)
                print(file_name)
                file_path = os.path.join("././data/", file_name)
                print("filepath= " + file_path)
                file.stream.seek(0)
                file.save(file_path)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("file not saved")
                return create_response("FileSaveError", "")       
            
            # upload image on Azure
            try:
                trn_azure_folder_name = os.environ['TRN_AZURE_FOLDER_NAME']
                image_url, status = upload_to_azure(file_path, trn_azure_folder_name)
                print("image_url = " + str(image_url))
            except Exception as ex:
                print("error msg = " + str(ex))
                print("file not uploaded")
                return create_response("FileSaveError", "")             

            # update image url details into db
            try:
                image_data = {}
                image_data["id"] = image_id
                image_data["image_url"] = image_url
                update_image_in_db(image_data)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("image data not updated in database") 
                return create_response("DatabaseUpdateError", "")           
                    
        else:
            return create_response("APIRequestIncorrectDataError", "")

    # Update response in database
    try:
        response_data = {}
        response_data["api_request_id"] = request_id
        response_data["api_response"] = resp
        update_response_in_db(response_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("response not updated in database")
        return create_response("DatabaseUpdateError", "")

    # Prepare & return response
    return create_response("Success", resp)