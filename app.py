import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


from flask import Flask, request, render_template
import pickle


app = Flask("__name__")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

model = pickle.load(open("telecom_churn_best_model.sav","rb"))

q = ""

@app.route("/")
def loadPage():
      return render_template("home.html", query="")


@app.route("/",methods=["POST"])
def predict():
    
      '''
      Contract_Month-to-month
      TotalCharges
      OnlineSecurity_No
      TechSupport_No
      OnlineBackup_No
      Contract_Two year
      PaymentMethod_Electronic check
      MonthlyCharges
      DeviceProtection_No
      tenure_group_1 - 12
    '''
      inputQuery1 = request.form['query1']
      inputQuery2 = request.form['query2']
      inputQuery3 = request.form['query3']
      inputQuery4 = request.form['query4']
      inputQuery5 = request.form['query5']
      inputQuery6 = request.form['query6']
      inputQuery7 = request.form['query7']
      inputQuery8 = request.form['query8']
      inputQuery9 = request.form['query9']
      inputQuery10 = request.form['query10']


      data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10]]

      input_df = pd.DataFrame(data, columns = ["Contract_Month-to-month", "TotalCharges", "OnlineSecurity_No", "TechSupport_No", "OnlineBackup_No",
      "Contract_Two year", "PaymentMethod_Electronic check", "MonthlyCharges", "DeviceProtection_No", "tenure_group_1 - 12"])

      single = model.predict(input_df)
      proba = model.predict_proba(input_df)[:,1][0]
      proba2 = model.predict_proba(input_df)[:,0][0]

      if single ==1:
            o1 = "This customer is likely to CHURN!!"
            o2 = f"Confidence level equals to {(proba*100):.2f}%"
      else:
            o1 = "This customer is likely to CONTINUE!!"
            o2 = f"Confidence level equals to {(proba2*100):.2f}%"

      return render_template('home.html', output1=o1, output2=o2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'])       
                                      
app.run(debug = True)