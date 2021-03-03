import numpy as np 
import os 
from flask import Flask, render_template, request
import pickle 


app = Flask(__name__)
model = pickle.load(open('models/LogisticClassifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    def decode(x):
        if x == 0:
            return 'Iris-setosa'
        elif x == 1:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'
    int_features = [float(x) for x in request.form.values()]
    final_features= [ np.array(int_features) ]
    prediction = model.predict(final_features)
    result = prediction[0]
    return render_template('index.html',prediction_text = 'Class of the Flower is : {}'.format(decode(result)))

if __name__ == "__main__":
    app.run(debug=True)