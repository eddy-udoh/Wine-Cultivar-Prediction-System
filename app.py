from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model components
def initialize_model():
    with open('model.h5', 'rb') as file:
        components = pickle.load(file)
        return components['model'], components['scaler'], components['target_names']

classifier, normalizer, class_labels = initialize_model()

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def classify_wine():
    try:
        input_json = request.get_json()
        
        # Extract all 13 chemical features
        feature_array = np.array([[
            input_json['alcohol'],
            input_json['malic_acid'],
            input_json['ash'],
            input_json['alkalinity'],
            input_json['magnesium'],
            input_json['phenols'],
            input_json['flavanoids'],
            input_json['nonflavanoid'],
            input_json['proanthocyanins'],
            input_json['color'],
            input_json['hue'],
            input_json['dilution'],
            input_json['proline']
        ]])
        
        # Normalize and classify
        normalized_features = normalizer.transform(feature_array)
        predicted_class = classifier.predict(normalized_features)[0]
        class_probabilities = classifier.predict_proba(normalized_features)[0]
        
        max_probability = float(np.max(class_probabilities))
        
        return jsonify({
            'predicted_class': int(predicted_class) + 1,
            'confidence': max_probability,
            'probabilities': [float(p) for p in class_probabilities],
            'status': 'success'
        })
    
    except Exception as error:
        return jsonify({
            'error': str(error),
            'status': 'error'
        }), 400

if __name__ == "__main__":
    app.run(debug=True)