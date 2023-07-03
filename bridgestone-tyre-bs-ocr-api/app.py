from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from src.helpers.utils import create_response
from src.ocr.tsn import get_ocr_tsn
from src.ocr.trn import get_ocr_trn
from src.ocr.validate_trn import validate_ocr_trn
from src.ocr.validate_tsn import validate_ocr_tsn

app = Flask(__name__)
cors = CORS(app)
app.secret_key = os.environ['APP_SECRET']

@app.route('/tsn', methods=['POST'])
@cross_origin()
def tsn_ocr():
    try:
        return get_ocr_tsn(request)        
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from tsn_ocr")
        return create_response("InternalServerError", "") 

@app.route('/trn', methods=['POST'])
@cross_origin()
def trn_ocr():
    try:
        return get_ocr_trn(request)        
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from tsn_ocr")
        return create_response("InternalServerError", "") 

@app.route('/validate_trn', methods=['POST'])
@cross_origin()
def validate_trn():
    try:
        return validate_ocr_trn(request)
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from validate_ocr_trn")
        return create_response("InternalServerError", "")

@app.route('/validate_tsn', methods=['POST'])
@cross_origin()
def validate_tsn():
    try:
        return validate_ocr_tsn(request)
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from validate_ocr_tsn")
        return create_response("InternalServerError", "")

if __name__ == "__main__":
    app.run(debug=True, port=5002)

