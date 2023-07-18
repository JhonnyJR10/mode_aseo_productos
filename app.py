import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar las predicciones desde el archivo
with open('predicciones.pckl', 'rb') as file:
    predictions = pickle.load(file)

@app.route("/katana-ml/api/v1.0/forecast/ironsteel", methods=['POST'])
def predict():
    horizon = int(request.json['horizon'])
    producto = request.json['producto']
    
    # Obtener las predicciones correspondientes al producto
    if producto in predictions:
        forecast = predictions[producto]
        data = forecast.tail(horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    else:
        data = pd.DataFrame()  # Producto no encontrado
    
    ret = data.to_json(orient='records', date_format='iso')
    
    return ret

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run()