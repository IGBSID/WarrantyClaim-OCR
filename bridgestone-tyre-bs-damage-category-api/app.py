from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
from src.validate.validate_damage_category import validate
from src.damage_category.damage_category import fetch_damage_category, damage_category_response
from src.helpers.utils import create_response
#from gevent.pywsgi import WSGIServer
app = Flask(__name__)
cors = CORS(app)

app.secret_key = os.environ['APP_SECRET']

@app.route('/damage_category', methods=['POST'])
@cross_origin()
def damage_category():
    try:
        return fetch_damage_category(request)        
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from damage_category")
        return create_response("InternalServerError") 

@app.route('/damage_category_response', methods=['POST'])
@cross_origin()
def other_damage_category_response():
    try:
        return damage_category_response(request)        
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from damage_category")
        return create_response("InternalServerError", "") 


@app.route('/validate', methods=['POST'])
@cross_origin()
def validate_damage_category():
    try:
        return validate(request)
    except (Exception) as ex:
        print("error msg = " + str(ex))
        print("Error from validate_damage_category")
        return create_response("InternalServerError", "")

if __name__ == "__main__":
     app.run(debug=True, port=5000)
#    http_server = WSGIServer(('0.0.0.0', 5000), app)
#    http_server.serve_forever()

