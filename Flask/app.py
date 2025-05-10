from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, jsonify
from markupsafe import Markup
from model import predict_image
import utils

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            res = Markup(utils.disease_dic[prediction])
            return render_template('display.html', status=200, result=res)
        except:
            pass
    return render_template('index.html', status=500, res="Internal Server Error")

@app.route('/logout', methods = ['GET','POST'])
def logout():
    if request.method == 'POST':
        try:
            return redirect("http://127.0.0.1:7000")
        except:
            pass



if __name__ == "__main__":
    app.run(debug=True)
