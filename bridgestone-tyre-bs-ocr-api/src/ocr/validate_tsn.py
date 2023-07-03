from src.helpers.error import TokenError, APIRequestIncorrectDataError
from src.helpers.auth import authenticate
from src.helpers.utils import create_response, update_validate_request_in_db

def validate_ocr_tsn(request):
    try:
        # get request data
        request_data = {}
        request_data["record_id"] = request.form.get('Record_Id')
        request_data["sent_on"] = request.form.get('Sent_On')
        request_data["image_details"] = request.form.get('Image_Details')
        request_data["correct_ocr"] = request.form.get('Correct_Ocr')
        request_data["is_ocr_correct"] = request.form.get('Is_Ocr_Correct')
        print("request.files=" + str(request.files))
        print("request.form=" + str(request.form))
        
        if  (request is None or  # no request
            request_data["record_id"] is None or 
            request_data["sent_on"] is None or \
            request_data["image_details"] is None or \
            request_data["is_ocr_correct"] is None):
            print("some error")
            raise APIRequestIncorrectDataError
    except (APIRequestIncorrectDataError, Exception) as ex:
        print("error msg = " + str(ex))
        print("API request data not correct")
        return create_response("APIRequestIncorrectDataError", "") 

    # check if feedback is given in case the code is not correct
    try:
        if request_data["is_ocr_correct"] == "1":
            if request_data["correct_ocr"] is None:
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

    # update request into db
    try:
        request_id = update_validate_request_in_db(request_data, "tsn")
        print("request_id = " + str(request_id))
    except Exception as ex:
        print("error msg = " + str(ex))
        print("API request data not updated in database")
        return create_response("DatabaseUpdateError", "")   

    return create_response("Success", "")        