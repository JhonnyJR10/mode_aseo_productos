import pickle
import pandas as pd
with open('forecast_aseo.pckl', 'rb') as fin:
    m2 = pickle.load(fin)

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)
@app.route("/katana-ml/api/v1.0/forecast/ironsteel", methods=['POST'])
def predict():
    horizon = int(request.json['horizon'])
    
    future2 = m2.make_future_dataframe(periods=horizon, freq='M')
    forecast2 = m2.predict(future2)
    
    data = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-horizon:]
    
    ret = data.to_json(orient='records', date_format='iso')
    
    return ret
# running REST interface, port=3000 for direct test
if __name__ == "__main__":
    app.run()