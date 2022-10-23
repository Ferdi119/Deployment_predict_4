'''
	Contoh Deloyment untuk Domain Data Science (DS)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load
from IPython.core.display import HTML

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')
model = None


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

        # return render_template("index.html", data=X_test.to_html(index=False, escape=False, formatters=dict(gambar=to_img_tag)))


# [Routing untuk API]
@ app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Nilai default untuk variabel input atau features (X) ke model
    input_sepal_length = 5.1
    input_sepal_width = 3.5
    input_petal_length = 1.4
    input_petal_width = 0.2

    if request.method == 'POST':
        # Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
        input_sepal_length = float(request.form['sepal_length'])
        input_sepal_width = float(request.form['sepal_width'])
        input_petal_length = float(request.form['petal_length'])
        input_petal_width = float(request.form['petal_width'])

        # Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
        df_test = pd.DataFrame(data={
            "SepalLengthCm": [input_sepal_length],
            "SepalWidthCm": [input_sepal_width],
            "PetalLengthCm": [input_petal_length],
            "PetalWidthCm": [input_petal_width]
        })

        hasil_prediksi = model.predict(df_test[0:1])[0]

        # Set Path untuk gambar hasil prediksi
        if hasil_prediksi == 'Iris-setosa':
            gambar_prediksi = '/static/images/iris_setosa.jpg'
        elif hasil_prediksi == 'Iris-versicolor':
            gambar_prediksi = '/static/images/iris_versicolor.jpg'
        else:
            gambar_prediksi = '/static/images/iris_virginica.jpg'

        # Return hasil prediksi dengan format JSON
        return jsonify({
            "prediksi": hasil_prediksi,
            "gambar_prediksi": gambar_prediksi
        })

# @app.route("/api/deteksi2",methods=['POST'])
# def index():
# 	if request.method == "POST":
# 		csv_file = request.files.get("file")
# 		X_test = pd.read_csv(csv_file)
# 		X_test["prediksi"] = model.predict(X_test)
# 		return render_template("index.html", data=X_test.to_json()
# =[Main]========================================


if __name__ == '__main__':

    # Load model yang telah ditraining
    model = load('model_iris_dt.model')

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
