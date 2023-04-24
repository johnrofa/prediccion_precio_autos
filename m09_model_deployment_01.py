#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_proba(url):

    try:
    
        regRF11 = joblib.load(os.path.dirname(__file__) + '/phishing_clf_01.pkl') 
        leMake= joblib.load(os.path.dirname(__file__) + '/leMake_01.pkl')
        leModel= joblib.load(os.path.dirname(__file__) + '/leModel_01.pkl')
        leState= joblib.load(os.path.dirname(__file__) + '/leState_01.pkl')  


        url_ = pd.DataFrame([url], columns=['url'])
        domain = url_.url.str.split('/', expand=True)
        domain.rename(columns={3: "Year", 4: "Mileage", 5: "State", 6: "Make", 7:"Model"}, inplace=True)
        domain_01=domain.drop([0, 1, 2], axis=1)
        domain_01['ID']=0
        domain_02=domain_01.set_index('ID')
    
        #Transformación
    
        domain_02["State"]=leState.transform(domain_02.State)
        domain_02["Make"]=leMake.transform(domain_02.Make)
        domain_02["Model"]=leModel.transform(domain_02.Model)    

    
        # Make prediction
        ypredRF11 = regRF11.predict(domain_02)
    except:
        ypredRF11= -999
    
    
    return ypredRF11


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        ypredRF11 = predict_proba(url)
        
        print(url)
        print('Costo del vehiculo: ', ypredRF11)
        