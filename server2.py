from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

workpath = os.path.dirname(os.path.abspath(__file__))
vehiclefile = os.path.join(workpath, 'vehiclesFinal.csv')
scalerfile = os.path.join(workpath, 'StandardScaler.sav')
modelfile = os.path.join(workpath, 'XGBoostDeploy.sav')

df = pd.read_csv(vehiclefile)
cat_cols=['manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','state']
temp = {}
for i in cat_cols:
    temp[i] = df[i].unique().flatten()

@app.route('/')
def index():
    return render_template('index.html', data=temp)


@app.route('/predict', methods=['POST'])
def predict():
    standardscaler = pickle.load(open(scalerfile, 'rb'))
    mymodel = pickle.load(open(modelfile, 'rb'))
    year = int(request.form.get("year"))
    odometer = int(request.form.get("odometer"))
    data = [[year, odometer]]

    columns = ['year', 'odometer']
    year_odometer = pd.DataFrame(data=data, columns=columns)

    # Transform the data using the loaded StandardScaler
    year1_scaled = standardscaler.transform(year_odometer[['year']])  # Transform using both 'year' and 'odometer'
    odometer2_scaled = standardscaler.transform(year_odometer[['odometer']])
    year1 = year1_scaled[0][0]
    odometer2 = odometer2_scaled[0][0]
    testcols = ['year', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status',
                'transmission', 'drive', 'type', 'paint_color', 'state']
    testdata = [year1, int(request.form['manf']), int(request.form['MODEL']),
                int(request.form['condition']),
                int(request.form['cylinders']), int(request.form['fuel']), odometer2,
                int(request.form['tstatus']),
                int(request.form['transmission']), int(request.form['drive']),
                int(request.form['type']), int(request.form['paint_color']), int(request.form['state'])]

    test = pd.DataFrame(data=[testdata], columns=testcols, dtype=None)
    pred = mymodel.predict(test)
    price = (np.exp(pred[0]))*2.5
    temp['price'] = price

    return render_template(
        "result.html",data=temp)
if __name__ == '__main__':
    app.run(debug=True)
