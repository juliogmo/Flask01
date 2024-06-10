from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modeloRF.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        battery_power = float(request.form['battery_power'])
        int_memory = float(request.form['int_memory'])
        mobile_wt = float(request.form['mobile_wt'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        ram = float(request.form['ram'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[battery_power, int_memory, mobile_wt, px_height, px_width, ram]], 
                               columns=['battery_power', 'int_memory', 'mobile_wt', 'px_height', 'px_width', 'ram'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a tipo de datos estándar de Python
        prediction_result = int(prediction[0])
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction_result})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)