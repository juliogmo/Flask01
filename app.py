from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('RandomForest2.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        non_urgent_order = float(request.form['non_urgent_order'])
        urgent_order = float(request.form['urgent_order'])
        order_type_b = float(request.form['order_type_b'])
        order_type_c = float(request.form['order_type_c'])
        
        # Crear un DataFrame con los datos usando los nombres de características correctos
        data_df = pd.DataFrame(
            [[non_urgent_order, urgent_order, order_type_b, order_type_c]], 
            columns=['Non-urgent order', 'Urgent order', 'Order type B', 'Order type C']
        )
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a tipo de datos estándar de Python
        prediction_result = float(prediction[0])
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediccion': prediction_result})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
