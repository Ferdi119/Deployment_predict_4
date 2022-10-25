from flask import Flask,render_template,request,jsonify, Response
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        SepalLengthCm = float(request.form['SepalLengthCm'])
        SepalWidthCm = float(request.form['SepalWidthCm'])
        PetalLengthCm = float(request.form['PetalLengthCm'])
        PetalWidthCm = float(request.form['PetalWidthCm'])


        values = np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
        prediction = model.predict(values)


        return render_template('index.html', prediction_text='Hasilnya adalah {}'.format(prediction))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        csv_file = request.files.get("file")
        X_test = pd.read_csv(csv_file)
        X_test["prediksi"] = model.predict(X_test)
        pd.set_option("display.precision", 2)
        setosa = '/static/images/iris_setosa.jpg'
        versi_color = '/static/images/iris_versicolor.jpg'
        virginica = '/static/images/iris_virginica.jpg'
        lis = []
        for i in X_test['prediksi']:
            if i == 'Iris-setosa':
                lis.append(setosa)
            elif i == 'Iris-versicolor':
                lis.append(versi_color)
            else:
                lis.append(virginica)
        X_test['gambar'] = lis

        def to_img_tag(path):
            return '<img src="' + path + '" width="100" >'
        # return render_template("index.html", tables=[X_test.to_html(classes='table table-stripped', index=False, escape=False, formatters=dict(gambar=to_img_tag))], titles=[''])
        return render_template("index.html", tables=[X_test.to_html(classes='table table-stripped', index=False, escape=False, formatters=dict(gambar=to_img_tag)).replace('border="1"','border="0"')], titles=[''])

# =[Main]========================================

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = load('model_iris_dt.model')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)