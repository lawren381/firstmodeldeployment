from flask import Flask, render_template, request
from keras import model_from_json
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
import joblib
import tensorflow as tf

global models


def init():
    json_file = open('model_train.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_train.h5")
    print("Loaded Model from disk")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model


app = Flask(__name__, template_folder="template")


@app.route('/')
def home():
    return render_template('index.html')


'''
def valuepredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 13)
    loaded_model = pickle.load(open("ann.pkl", "rb"))
    predict = loaded_model(to_predict)
    return predict[0]'''


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        pred_args = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        features = np.array(pred_args).reshape(1,-1)
        models = init()
        predicted = models.predict(features)

    if predicted == 1:
        res = "likely"
    else:
        res = "not likely"
    # return res
    return render_template('finalprediction.html', prediction=res)


if __name__ == '__main__':
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
