import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route("/katana-ml/api/v1.0/forecast/ironsteel", methods=['POST'])
def predict():
    horizon = int(request.json['horizon'])
    
    future2 = predicciones_cargadas.make_future_dataframe(periods=horizon, freq='M')
    forecast2 = predicciones_cargadas.predict(future2)
    
    data = forecast2[['producto', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']][-horizon:]
    
    ret = data.to_json(orient='records', date_format='iso')
    
    return ret

# Load the Prophet model from pickle file
with open('predicciones.pckl', 'rb') as fin:
    predicciones_cargadas = pickle.load(file)

# running REST interface, port=3000 for direct test
if __name__ == "__main__":
    app.run()