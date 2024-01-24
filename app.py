import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
knn_loaded = joblib.load('./static/knn.joblib')
etr_loaded = joblib.load('./static/etr.joblib')
br_loaded = joblib.load('./static/br.joblib')
ar_loaded = joblib.load('./static/ar.joblib')
xg_loaded = joblib.load('./static/xg.joblib')
lr_loaded = joblib.load('./static/lr.joblib')
rf_loaded = joblib.load('./static/rf.joblib')
gbr_loaded = joblib.load('./static/gbr.joblib')
dtr_loaded = joblib.load('./static/dtr.joblib')
svr_loaded = joblib.load('./static/svr.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features=[]
    for x in request.form.values():
        try:
            int_features.append(int(float(x)))
        except ValueError:
            # Handle the case where conversion is not possible
            print(f"Skipping invalid value: {x}")

    # Now int_features contains the successfully converted integer values
    print(int_features)
    # int_features = [int(x) for x in request.form.values()]
    mo= int_features[0]
    prediction=[]
    int_features= int_features[1:]
    final_features = [np.array(int_features)]
    if mo==1:
        prediction = knn_loaded.predict(final_features)
    elif mo==2:
        prediction = etr_loaded.predict(final_features)
    elif mo==3:
        prediction=br_loaded.predict(final_features)
    elif mo==4:
        prediction= ar_loaded.predict(final_features)
    elif mo==5:
        ff = np.array(int_features)
        prediction = xg_loaded.predict(ff.reshape(1,-1))
    elif mo==6:
        prediction= lr_loaded.predict(final_features)
    elif mo==7:
        prediction= rf_loaded.predict(final_features)
    elif mo==8:
        prediction= dtr_loaded.predict(final_features)
    elif mo==9:
        prediction= svr_loaded.predict(final_features)
    else:
        prediction= gbr_loaded.predict(final_features)
        

    return render_template('temp.html', prediction_text='The predicted Compressive Strength of concrete is {} kN/m3'.format(round(prediction[0],2)))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)