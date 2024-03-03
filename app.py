from flask import Flask,render_template,request
import cv2
import pickle
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    img = request.files['file']

    upload_folder = os.path.join(os.getcwd(), 'uploads')  # 'uploads' folder in the same directory
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, img.filename)
    img.save(file_path)
    
    img = cv2.imread(file_path)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resizedimg = cv2.resize(grayimg,(28,28))
    newimg = tf.keras.utils.normalize(resizedimg, axis=1)
    newimg = np.array(newimg).reshape(-1,28,28,1)
    predictions = model.predict(newimg)
    prediction = np.argmax(predictions, axis=1)
    
    return render_template('index.html', prediction_text='Digit is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug = True)