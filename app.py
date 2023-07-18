import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

# Cargar las predicciones desde el archivo
with open('predicciones.pckl', 'rb') as file:
    predicciones_cargadas = pickle.load(file)

@app.route("/katana-ml/api/v1.0/forecast/ironsteel", methods=['POST'])
def predict():
    # Obtener los datos de entrada
    data = request.json['data']
    horizon = int(request.json['horizon'])
    
    future = predicciones_cargadas.make_future_dataframe(periods=horizon, freq='M')
    forecast = predicciones_cargadas.predict(future)
    
    # Filtrar los datos para los productos específicos
    data_filtered = forecast[forecast['producto'].isin(data)]
    
    # Obtener las últimas 'horizon' filas
    data_filtered = data_filtered.tail(horizon)
    
    # Seleccionar las columnas requeridas
    data_selected = data_filtered[['producto', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Convertir a formato JSON
    ret = data_selected.to_json(orient='records', date_format='iso')
    
    return ret

# running REST interface, port=3000 for direct test
if __name__ == "__main__":
    app.run()