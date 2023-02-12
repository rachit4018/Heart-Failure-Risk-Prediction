
from urllib import request
import joblib as jb
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import json
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# Metrics and preprocessing
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# TF and Keras
#import tensorflow as tf
import keras
import model

app = Flask(__name__)
ap = ""
name = ""
#y_train =jb.load('y_train.pkl')
y_test = jb.load('y_test.pkl')
accuracy = ""
y_test_predict = ""
@app.route("/", methods=["GET", "POST"])
def Fun_knn():
    return render_template("index.html")


@app.route("/sub", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        #int_features = model.preprocess(pd.DataFrame.from_dict([input_dict]))
        predictions = model.prediction(pd.DataFrame.from_dict([input_dict]))                       
            #accuracy = accuracy_score(y_test1,predictions)
        print(predictions)
        output = predictions[0]
        print(output)
        if output < 0.5:
            return render_template("sub.html", prediction_text="No chance of Heart Failure")
        else:
            return render_template("sub.html", prediction_text="Sorry, you are at risk of Heart Failure")

if __name__=="__main__":
    app.run(debug=True) 
