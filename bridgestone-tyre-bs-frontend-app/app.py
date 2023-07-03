import json
import os
import requests
from flask import Flask, flash, redirect, render_template, request, url_for,jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import datetime

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = "cairocoders-ednalan"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
UPLOAD_FOLDER = 'static/data/uploaded_images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home_page():
    return render_template('home.html')
     
@cross_origin()     
@app.route('/damage')
def upload_form():
    return render_template('damage_category.html',damage_result =[],list_result =[])

@cross_origin()
@app.route('/damage', methods=['POST'])
def upload_image():
    if 'Image_List' not in request.files:
        flash('No file part')
        return redirect(request.url)
    # files = request.files.getlist('Image_List')
    # No_Of_Images = request.form.get('No_Of_Images')
    # Record_Id = request.form.get('Record_Id')
    # Sent_On = request.form.get('Sent_On')
    # Image_Details = request.form.get('Image_Details')
    files = request.files.getlist('Image_List')
    No_Of_Images = 2
    Record_Id = 23456
    Sent_On = "2022-12-05 17:42:34"
    Image_Details = "damage image"
    print("Data", files)
    print("No_Of_Images", No_Of_Images)
    print("Record_Id",Record_Id)
    print("Sent_On",Sent_On)
    print("Image_Details",Image_Details)
    file_names = []
    f_result = []
    damage_result = []
    list_result = []
    data = {}
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            files={'Image_List': open(filepath,'rb'), 'Content-Type': 'image/png'}
            input_data = {"No_Of_Images": No_Of_Images,"Record_Id": Record_Id,"Sent_On":Sent_On,"Image_Details":Image_Details}
            Headers = { "Http-X-Api-Secret": "31cb52c2-7b79-11ed-a1eb-0242ac120002" }
            url = "https://bridgestone-damage-category-api.azurewebsites.net/damage_category"
            # url = "http://127.0.0.1:5001/damage_category"
            response = requests.post(url,data = input_data, files=files,headers=Headers)
            json_data = json.loads(response.text)
            damage_category = response.text
            f_result.append(json_data['damage_category_name'])
           
            # print("json_data", json_data)
            pred_category = json_data['damage_category_name']
            if pred_category not in data:
                data[pred_category] = json_data['other_damage_category']
            else:
                others = data[pred_category]
                data[pred_category] = others + json_data['other_damage_category']
             
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    
    final_category = max(set(f_result), key = f_result.count)
    data_final_category = data[final_category]
    if data_final_category:
        other_damages_merged = {}
        for other_cate in data_final_category:
            sub_cate = other_cate['damage_name']
            if sub_cate not in other_damages_merged:
                other_damages_merged[sub_cate] = [other_cate['priority']]
            else:
                other_damages_merged[sub_cate].append(other_cate['priority'])
        # print("other_damages_merged:", other_damages_merged)
        other_damages_merged_prioritised = {}
        for cate, priority_list in other_damages_merged.items():
            other_damages_merged_prioritised[cate] = max(set(priority_list), key=priority_list.count)
        # print("other_damages_merged_prioritised", other_damages_merged_prioritised )
        for category, priority in other_damages_merged_prioritised.items():
            category_and_priority = f"{category} ({priority})"
            # if category_and_priority not in list_result:
            list_result.append(category_and_priority)

    return render_template('damage_category.html', filenames=file_names,damage_result =damage_result,final_category=final_category,list_result=list_result)
    

@cross_origin()
@app.route('/damage/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='data/uploaded_images/' + filename), code=301)
    # return jsonify(filename)

@cross_origin()
@app.route('/static/uploads/<filename>')
def display_damage(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='data/uploaded_images/' + filename), code=301)

@cross_origin()
@app.route('/home')
def cancelRecord():
    return render_template('/damage_category.html', filenames=[],damage_result=[],list_result = [])
    # return jsonify("hello")

@cross_origin()
@app.route('/tsn',methods=['POST'])
def tsn_image():
    if 'Image_List' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('Image_List')
    No_Of_Images = 1
    Record_Id = 23456
    Sent_On = "2022-12-05 17:42:34"
    Image_Details = "tsn image"
    print("Data", files)
    print("No_Of_Images",No_Of_Images)
    print("Record_Id",Record_Id)
    print("Sent_On",Sent_On)
    print("Image_Details",Image_Details)
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            input_data = {"No_Of_Images": No_Of_Images,"Record_Id": Record_Id,"Sent_On":Sent_On,"Image_Details":Image_Details}
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            files={'Image_List': open(filepath,'rb'), 'Content-Type': 'image/png'}        
            url1 = "https://bridgestone-ocr-api.azurewebsites.net/tsn"
            Headers = { "Http-X-Api-Secret": "31cb52c2-7b79-11ed-a1eb-0242ac120002" }
            resp = requests.post(url1, data=input_data,files=files,headers=Headers)
            json_respt=json.loads(resp.text)
            ocr_result = json_respt['ocr_result']
            json_resp = json.loads(resp.text)
            return json_resp
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('tsn_ocr.html',filenames=file_names,ocr_result=ocr_result)
    # return jsonify(data)

@cross_origin()
@app.route('/tsn_home')
def cancelTsnRecord():
    return render_template('/tsn_ocr.html', filenames=[])

@cross_origin()
@app.route('/trn',methods=['POST'])
def trn_image():
    if 'Image_List' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('Image_List')
    No_Of_Images = 1
    Record_Id = 23456
    Sent_On = "2022-12-05 17:42:34"
    Image_Details = "tsn image"
    print("Data", files)
    print("No_Of_Images",No_Of_Images)
    print("Record_Id",Record_Id)
    print("Sent_On",Sent_On)
    print("Image_Details",Image_Details)
    file_names = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_names.append(filename)
            input_data = {"No_Of_Images": No_Of_Images,"Record_Id": Record_Id,"Sent_On":Sent_On,"Image_Details":Image_Details}
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            files={'Image_List': open(filepath,'rb'), 'Content-Type': 'image/png'}        
            url1 = "https://bridgestone-ocr-api.azurewebsites.net/trn"
            Headers = { "Http-X-Api-Secret": "31cb52c2-7b79-11ed-a1eb-0242ac120002" }
            resp = requests.post(url1, data=input_data,files=files,headers=Headers)
            json_respt=json.loads(resp.text)
            ocr_result = json_respt['ocr_result']
            json_resp = json.loads(resp.text)
            return json_resp
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('trn_ocr.html',filenames=file_names,ocr_result=ocr_result)

@cross_origin()
@app.route('/trn_home')
def cancelTrnRecord():
    return render_template('/trn_ocr.html', filenames=[])
    
@cross_origin()
@app.route('/tsn')
def tsn():
    return render_template('tsn_ocr.html')

@cross_origin()
@app.route('/trn')
def trn():
    return render_template('trn_ocr.html')

@cross_origin()
@app.route('/damage')
def damage():
    return render_template('damage_category.html')

  
if __name__ == "__main__":
    app.run(debug=True, port=5000)