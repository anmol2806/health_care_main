#app.py
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib

app = Flask(__name__)
# load the saved model file and use for prediction
logit_model = joblib.load('logit_model.pkl')
logit_model_diabetes = joblib.load('logit_diabetes_model.pkl')
logit_model_bmi=joblib.load(open('clf.pkl','rb'))

@app.route('/')
def home():
    return render_template("index_main.html")




@app.route('/heart')
def home1():
    return render_template("index.html")

@app.route('/heart/predict',methods=['POST','GET'])
def predict1():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
       
    prediction1=logit_model.predict(final_features)
    
   
    return render_template('preview.html', pred= prediction1)
    
    
    
    
    
@app.route('/diabetes')
def home2():
    return render_template('index_diabetes.html')  

@app.route('/diabetes/predict',methods=['POST'])
def predict2():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features1 = [np.array(float_features)]
    prediction2 = logit_model_diabetes.predict(final_features1 )

    if prediction2 == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction2 == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index_diabetes.html', prediction_text='{}'.format(output))
    





@app.route('/bmi')
def home3():
    return render_template("index_bmi.html")


@app.route('/bmi/predict',methods=['POST','GET'])
def predict3():
    int_features = [int(x) for x in request.form.values()]
    y=int_features[2]/(int_features[1]*0.0254)**2
    int_features.append(y)
    final_features = [np.array(int_features)]
    prediction=logit_model_bmi.predict(final_features)
    
    return render_template('index_bmi.html', pred=prediction)




if __name__ == '__main__':
    app.run(debug=False)
