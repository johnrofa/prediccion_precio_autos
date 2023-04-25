#!/usr/bin/python

from flask import Flask,  request, jsonify
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment_01 import predict_proba
from flask_cors import CORS
import pandas as pd
import sys
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Prediccion Precio Vehiculos',
    description='Prediccion Precio Vehiculos')

ns = api.namespace('predict', 
     description='Prediccion Precio Vehiculos')
   
parser = api.parser()

parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@app.route('/modelo1', methods=["POST"])
def addUser():
    print(request.data)
    body = json.loads(request.data)
    print(body)

    # Leer los datos post
    Year = body["Year"]
    Mileage = body["Mileage"]
    State = body["State"]
    Make = body["Make"]
    Model = body["Model"]
    ID=0


    #Year = 2017
    #Mileage = 5362
    #State = " WI"
    #Make = "Jeep"
    #Model = "Wrangler"
    #ID = 0

    columnas=['Year','Mileage', 'State', 'Make', 'Model', 'ID' ]
    datos = [[Year,Mileage, State, Make, Model, ID]]

    domain_01 = pd.DataFrame(datos, columns=columnas)
    domain_02 = domain_01.set_index('ID')

    print('domain_02 :', domain_02)
    print('ruta  :', os.path.dirname(__name__) +'phishing_clf_01.pkl')

    #Importar objetos -modelos
    regRF11 = joblib.load(os.getcwd() +'\phishing_clf_01.pkl')
    leMake= joblib.load(os.getcwd()+ '\leMake_01.pkl')
    leModel= joblib.load(os.getcwd() + '\leModel_01.pkl')
    leState= joblib.load(os.getcwd() + '\leState_01.pkl')

    domain_02["State"]=leState.transform(domain_02.State)
    domain_02["Make"]=leMake.transform(domain_02.Make)
    domain_02["Model"]=leModel.transform(domain_02.Model)

    # Make prediction
    ypredRF11 = regRF11.predict(domain_02)
    ypredRF11 =str(ypredRF11[0])
    print(ypredRF11)

    costovehiculo = {
        "El costo del vehiculo es ": ypredRF11,
    }

    return jsonify(costovehiculo), 200







@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['URL'])
        }, 200



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
