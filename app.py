from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("fish_weight_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[key]) for key in ['Length1', 'Length2', 'Length3', 'Height', 'Width']]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        return render_template('index.html', prediction_text=f'Predicted Fish Weight: {prediction:.2f}g')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


