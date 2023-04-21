from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import requests
import pickle
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3

# Define a flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")
        
@app.route('/index', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



model_path2 = 'models/model_xception.h5' # load .h5 Model
classes2 = {0:"Bacteria",1:"Fungi",2:"Nematodes",3:"Normal",4:"Virus"}
CTS = load_model(model_path2)
from keras.preprocessing.image import load_img, img_to_array
def model_predict2(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(224,224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
    #prediction = classes2[result]  
    
    if result == 0:
        return "Bacteria", "result.html"        
    elif result == 1:
        return "Fungi","result.html"
    elif result == 2:
        return "Nematodes","result.html"
    elif result == 3:
        return "Normal","result.html"
    elif result == 4:
        return "Virus","result.html"
    
    


@app.route('/predict2',methods=['GET','POST'])
def predict2():
    print("Entered")
    if request.method == 'POST':
        print("Entered here")
        file = request.files['file'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        #filename1 = os.path.join(UPLOAD_FOLDER, filename)

        print("@@ Predicting class......")
        pred, output_page = model_predict2(file_path,CTS)

        remdies = []

        if pred == 'Bacteria':
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `label` from data2 where `message` = ?",(pred,))
            remdies = cur.fetchall()
        
        elif pred == 'Fungi':
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `label` from data2 where `message` = ?",(pred,))
            remdies = cur.fetchall()

        elif pred == 'Nematodes':
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `label` from data2 where `message` = ?",(pred,))
            remdies = cur.fetchall()

        elif pred == 'Virus':
            val = 'viruses'
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `label` from data2 where `message` = ?",(val,))
            remdies = cur.fetchall()

        else:
            pred = pred
              
        return render_template(output_page, pred_output = pred, remdy = remdies, img_src=UPLOAD_FOLDER + file.filename)


    #this section is used by gunicorn to serve the app on Heroku

data1 = pd.read_csv("data/Crop_recommendation.csv", encoding= 'unicode_escape')

@app.route('/crop')
def crop():
    companies=sorted(data1['Rural Areas'].unique())
    
    return render_template('crop.html',companies=companies)

@app.route('/features')
def features():
    return render_template('features.html')

data = pd.read_csv("data/Crop_recommendation.csv", encoding= 'unicode_escape')
data = data[['N', 'P', 'K', 'temperature', 'humidity','ph','rainfall','Rural Areas','label']]

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
data['label']= label_encoder.fit_transform(data['label'])
data['Rural Areas']= label_encoder.fit_transform(data['Rural Areas'])
data['label'].unique()

x = data.iloc[:, 0:8]
y = data.iloc[:,8]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
predictions = RF.predict(x_test)

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form['N']
    P = request.form['P']
    K = request.form['K']
    T = request.form['T']
    H = request.form['H']
    P = request.form['P']
    R = request.form['R']
    R1 = request.form['R1']
    reg = request.form['1']

    if reg == 'Adilabad':
        reg1 = 1
    elif reg == 'Bhadradri':
        reg1 = 2
    elif reg == 'Bhadradri Kothagudem':
        reg1 = 3
    elif reg == 'Hyderabad':
        reg1 = 4
    elif reg == 'Jagtial':
        reg1 = 5
    elif reg == 'Kamareddy':
        reg1 = 6
    elif reg == 'Karimnagar':
        reg1 = 7
    elif reg == 'Khammam':
        reg1 = 8
    elif reg == 'Kothagudem':
        reg1 = 9
    elif reg == 'Mahabubnagar':
        reg1 = 10
    elif reg == 'Mancherial':
        reg1 = 11
    elif reg == 'Medak':
        reg1 = 12
    elif reg == 'Medchal':
        reg1 = 13
    elif reg == 'Nagarkurnool':
        reg1 = 14
    elif reg == 'Nalgonda':
        reg1 = 15
    elif reg == 'Nirmalll':
        reg1 = 16
    elif reg == 'Nizamabad':
        reg1 = 17
    elif reg == 'Peddapalli':
        reg1 = 18
    elif reg == 'Rangareddy':
        reg1 = 19
    elif reg == 'Sangareddy':
        reg1 = 20
    elif reg == 'Siddipet':
        reg1 = 21
    elif reg == 'Suryapet':
        reg1 = 22
    elif reg == 'Vikarabad':
        reg1 = 23
    elif reg == 'wanaparthy':
        reg1 = 24
    elif reg == 'Warangal':
        reg1 = 25
    elif reg == 'Warangal':
        reg1 = 26
    elif reg == 'Yadadri Bhuvangiri':
        reg1 = 27

    Soil_composition_list = np.array([N,P,K,T,H,P,R,reg1]).reshape(1,8)
    print(Soil_composition_list)
    
    crop = RF.predict(Soil_composition_list)
    print(crop)

    
    outcome = crop[0]
    if outcome == 0:
        predicted_crop = 'Rice'
        crop_link = 'https://en.wikipedia.org/wiki/Rice'
        crop_img = '../static/styles/images/rice.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )

    elif outcome == 1:
        predicted_crop = 'Rice'
        crop_link = 'https://en.wikipedia.org/wiki/Maize'
        crop_img = '../static/styles/images/maize.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 2:
        predicted_crop = 'Chickpea'
        crop_link = 'https://en.wikipedia.org/wiki/Chickpea'
        crop_img = '../static/styles/images/chikpea.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )

    elif outcome == 3:
        predicted_crop = 'Kidney Beans'
        crop_link = 'https://en.wikipedia.org/wiki/Rajma'
        crop_img = '../static/styles/images/kideneybeans.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 4:
        predicted_crop = 'Pigeon Peas'
        crop_link = 'https://en.wikipedia.org/wiki/Pigeon_pea'
        crop_img = '../static/styles/images/pigeonpea.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )

    elif outcome == 5:
        predicted_crop = 'Mothbeans'
        crop_link = 'https://en.wikipedia.org/wiki/Vigna_acconitifolia'
        crop_img = '../static/styles/images/mothbeans.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 6:
        predicted_crop = 'Mung Bean'
        crop_link = 'https://en.wikipedia.org/wiki/Mung_bean'
        crop_img = '../static/styles/images/mungbeans.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )

    elif outcome == 7:
        predicted_crop = 'Black Grams'
        crop_link = 'https://en.wikipedia.org/wiki/Vigna_mungo'
        crop_img = '../static/styles/images/blackgram.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 8:
        predicted_crop = 'Pomegranate'
        crop_link = 'https://en.wikipedia.org/wiki/Pomegranate'
        crop_img = '../static/styles/images/pomogranate.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )

    elif outcome == 9:
        predicted_crop = 'Banana'
        crop_link = 'https://en.wikipedia.org/wiki/Banana'
        crop_img = '../static/styles/images/banana.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 10:
        predicted_crop = 'Mango'
        crop_link = 'https://en.wikipedia.org/wiki/Mango'
        crop_img = '../static/styles/images/mango.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 11:
        predicted_crop = 'Grapes'
        crop_link = 'https://en.wikipedia.org/wiki/Grape'
        crop_img = '../static/styles/images/'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 12:
        predicted_crop = 'Watermelon'
        crop_link = 'https://en.wikipedia.org/wiki/Watermelon'
        crop_img = '../static/styles/images/watermelon.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 13:
        predicted_crop = 'Apple'
        crop_link = 'https://en.wikipedia.org/wiki/Apple'
        crop_img = '../static/styles/images/apple.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 14:
        predicted_crop = 'Orange'
        crop_link = 'https://en.wikipedia.org/wiki/Orange'
        crop_img = '../static/styles/images/orange.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 15:
        predicted_crop = 'Papaya'
        crop_link = 'https://en.wikipedia.org/wiki/Papaya'
        crop_img = '../static/styles/images/papaya.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 16:
        predicted_crop = 'Coconut'
        crop_link = 'https://en.wikipedia.org/wiki/Coconut'
        crop_img = '../static/styles/images/coconut.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 17:
        predicted_crop = 'Cotton'
        crop_link = 'https://en.wikipedia.org/wiki/Cotton'
        crop_img = '../static/styles/images/cotton.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 18:
        predicted_crop = 'Jute'
        crop_link = 'https://en.wikipedia.org/wiki/Jute'
        crop_img = '../static/styles/images/jute.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    elif outcome == 19:
        predicted_crop = 'Coffee'
        crop_link = 'https://en.wikipedia.org/wiki/Coffee'
        crop_img = '../static/styles/images/cofee.jpg'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


    else:
        predicted_crop = 'Tea'
        crop_link = 'https://en.wikipedia.org/wiki/Tea'
        crop_img = '../static/styles/images/'


        return render_template('prediction.html', predicted_crop = predicted_crop, crop_link = crop_link , crop_img = crop_img )


if __name__ == '__main__':
        app.run(debug=True)
   