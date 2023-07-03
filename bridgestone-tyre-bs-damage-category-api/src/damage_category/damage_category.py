from src.helpers.error import TokenError, errors, APIRequestIncorrectDataError, AdditionalFilesThanRequiredError, NonTyreImageError
from src.helpers.auth import authenticate
from src.model.inference import get_damage_category, get_damage_labels
from src.helpers.utils import max_category, create_response, insert_or_update_request_in_db, insert_image_in_db, upload_to_azure, update_image_in_db, update_response_in_db, get_category_code_from_db, delete_image_in_db,get_cascade_damage_labels, update_user_response_in_db,convert_sub_damage_category
import os
import itertools
from collections import Counter
import pandas as pd
import random
def priority_score(dictonary,Keymax):
    result=(dictonary.get(Keymax))
    value=[]
    for i in result:
        for j in i:
            value.append(j)
    df=pd.DataFrame(value).groupby(0).mean().reset_index()
    df[1]=df[1]*100/df[1].max()
    df=df.sort_values(1,ascending=False)
    res=df.values.tolist()
    dictonary[Keymax]=[res]
    return dictonary



def priority_level(dictonary,Keymax):
    result=(dictonary.get(Keymax))
    priority_name = ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]
    if len(result[0])>3:
        for i in range(len(result[0])):
            result[0][i][1]=priority_name[i]
    else:
        index = random.choice([0,1])
        result[0][0][1]=priority_name[index]
        result[0][1][1]=priority_name[index+1]
    return dictonary


def fetch_damage_category(request):
    try:
        # get request data
        request_data = {}
        result_data = []
        sub_categories = []
        sub_dc_max = []
        temp_list = []
        files = request.files.getlist('Image_List')
        request_data["no_of_images"] = request.form.get('No_Of_Images')
        request_data["record_id"] = request.form.get('Record_Id')
        request_data["sent_on"] = request.form.get('Sent_On')
        request_data["image_details"] = request.form.get('Image_Details')
        print("request.files=" + str(request.files))
        print("request.form=" + str(request.form))
        # print("request_data=" + str(request_data))
        # print("files=" + str(files))
                
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
            # file_count <= 8 or # more than 1 file is passed  
            request_data["no_of_images"] is None or \
            request_data["record_id"] is None or request_data["sent_on"] is None or \
            request_data["image_details"] is None):
            print("some error")
            raise APIRequestIncorrectDataError
    except (APIRequestIncorrectDataError, Exception) as ex:
        print("error msg = " + str(ex))
        print("API request data not correct")
        return create_response("APIRequestIncorrectDataError") 

    # check if more than allowed files are passed in input
    try:
        if file_count >= 8:
            raise AdditionalFilesThanRequiredError
    except (AdditionalFilesThanRequiredError, Exception) as ex:
        print("error msg = " + str(ex))
        print("More files sent than the maximum files allowed")
        return create_response("AdditionalFilesThanRequiredError") 
    
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
        return create_response("TokenError")           

    # insert request into db
    try:
        request_id, is_insert = insert_or_update_request_in_db(request_data)
        print("request_id = " + str(request_id))
    except Exception as ex:
        print("error msg = " + str(ex))
        print("API request data not added in database")
        return create_response("DatabaseInsertError")        

    if (not is_insert): #i.e. update, then delete all files
        try:
            image_data = {}
            image_data["api_request_id"] = request_id
            delete_image_in_db(image_data)
        except Exception as ex:
            print("error msg = " + str(ex))
            print("image data not deleted from database")
            return create_response("DatabaseDeleteError")

    D = {}
    
    for file in files:
        if file:
            
            # if (is_insert):
            # insert image details into db
            try:
                image_data = {}
                image_data["api_request_id"] = request_id
                # image_data["image_url"] = image_url
                image_data["image_details"] = request_data["image_details"]
                image_data["created_on"] = request_data["sent_on"]
                image_id = insert_image_in_db(image_data)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("image data not added in database")
                return create_response("DatabaseInsertError")
            # else:
            #     # get the image details
            
            # save the image locally
            try:
                file_extension = file.filename.split(".")[1]
                file_name = str(image_id)+"."+str(file_extension)
                print(file_name)
                file_path = os.path.join("././data/", file_name)
                print("filepath= " + file_path)
                file.save(file_path)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("file not saved")
                return create_response("FileSaveError")       
            
            # upload image on Azure
            try:
                image_url, status = upload_to_azure(file_path)
                print("image_url = " + str(image_url))
            except Exception as ex:
                print("error msg = " + str(ex))
                print("file not uploaded")
                return create_response("FileSaveError")      

            try:
                # get damage results from the model
                damage_result = get_damage_category(file_path)
                #sub_dc_max.append(damage_result[1])

                # handle non-tyre images
                try:
                    if len(damage_result[0]) <= 0:
                        raise NonTyreImageError                        
                except (Exception, NonTyreImageError) as ex:
                    print("error msg = " + str(ex))
                    print("Uploaded image is not a tyre image")
                    return create_response("NonTyreImageError") 
                print('*********************************************')     
                print(damage_result)
                print('*********************************************') 
                key = damage_result[1][0][0]
                if key in D.keys():
                    D[key].append(damage_result[1][1:])
                else:
                    D[key]=[]
                    D[key].append(damage_result[1][1:])

            except Exception as ex:
                print("error msg = " + str(ex))
                print("Unable to receive Damage model response")
                return create_response("DamageModelError")      

            # update image url details into db
            try:
                image_data = {}
                image_data["id"] = image_id
                image_data["image_url"] = image_url
                update_image_in_db(image_data)
            except Exception as ex:
                print("error msg = " + str(ex))
                print("image data not updated in database") 
                return create_response("DatabaseUpdateError")        
            
        else:
            return create_response("FileNotSentError")
        data =  get_damage_labels()
        #print(data)
        countingdict = {k:sum(1 for x in v if x) for k,v in D.items()}
        Keymax = max(zip(countingdict.values(), countingdict.keys()))[1]
        max_dc_value_index = data[0].index(Keymax)
        max_return_dc_value = data[1][max_dc_value_index]
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #print(D[Keymax])
        if len(D[Keymax][0])>=2:
            D1=priority_score(D.copy(),Keymax)
            print(D)
            print('****************************************')
            print(D1)
            D1= priority_level(D1,Keymax)
            sub_categories = D1[Keymax][0]
        else:
            
            sub_categories = D[Keymax][0]
        sub_categories_resp = []
        cascade_labels = get_cascade_damage_labels()
        
        # sub categories response using priority
        for sc in sub_categories:
            d= {}
            index_value = cascade_labels[0].index(sc[0])
            d["damage_name"]=cascade_labels[1][index_value]
            d["priority"]=sc[1]
            sub_categories_resp.append(d)
    # Update response in database
    try:
        response_data = {}
        response_data["api_request_id"] = request_id
        response_data["api_response"] = max_return_dc_value
        update_response_in_db(response_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("response not updated in database")
        return create_response("DatabaseUpdateError")

    # Prepare response
    damage_category_code = get_category_code_from_db(max_return_dc_value)

    return_message = errors["Success"]
    return_message["damage_category_code"] = damage_category_code
    return_message["damage_category_name"] = max_return_dc_value
    return_message["other_damage_category"] = sub_categories_resp
    print('###################################################################')
    #print(D1)
    print('###################################################################')
    return create_response("Success", return_message)


def damage_category_response(request):
    try:
        # get request data
        request_data = {}
        request_data["record_id"] = request.form.get('Record_Id')
        request_data["sent_on"] = request.form.get('Sent_On')
        # request_data["damage_category_code"] = request.form.get('Damage_Category_Code')
        request_data["damage_category_name"] = request.form.get('Damage_Category_Name')
        print("request.files=" + str(request.files))
        print("request.form=" + str(request.form))
        print("request.data=" + str(request_data))
        
        if  (request is None or  # no request
            request_data["record_id"] is None or 
            request_data["sent_on"] is None or \
            # request_data["damage_category_code"] is None or \
            request_data["damage_category_name"] is None):
            print("some error")
            raise APIRequestIncorrectDataError
    except (APIRequestIncorrectDataError, Exception) as ex:
        print("error msg = " + str(ex))
        print("API request data not correct")
        return create_response("APIRequestIncorrectDataError") 
    
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
        return create_response("TokenError")           

    # Update response in database
    try:
        # response_data = {}
        # # response_data["record_id"] = "damage_category"
        # response_data["api_response"] = max_return_value
        update_user_response_in_db(request_data)
    except Exception as ex:
        print("error msg = " + str(ex))
        print("response not updated in database")
        return create_response("DatabaseUpdateError")

    return create_response("Success")