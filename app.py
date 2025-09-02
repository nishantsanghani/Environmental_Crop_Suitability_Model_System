# -------------------------------
# Importing necessary libraries
# -------------------------------
from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn   # (kept here in case your model/scaler uses sklearn objects)

# -------------------------------
# Load trained model & scalers
# -------------------------------
# Ensure these files exist in your project directory
model = pickle.load(open('model.pkl', 'rb'))              # Trained ML model
sc = pickle.load(open('standscaler.pkl', 'rb'))           # StandardScaler
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))          # MinMaxScaler

# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Home route - renders input form
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')   # index.html should be inside /templates folder

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Collect input values from the form,
    scale them using MinMaxScaler + StandardScaler,
    predict the crop using trained ML model,
    and return the result.
    """

    # Get form values
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Convert values to numeric array (float type is important)
    feature_list = [float(N), float(P), float(K), float(temp), 
                    float(humidity), float(ph), float(rainfall)]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale features using loaded scalers
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Make prediction
    prediction = model.predict(final_features)

    # Dictionary mapping model outputs to crop names
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
        10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    crop_images = {
        "Rice": "rice.jpg",
        "Maize": "maize.jpg",
        "Jute": "jute.jpg",
        "Cotton": "cotton.jpg",
        "Coconut": "coconut.jpg",
        "Papaya": "papaya.jpg",
        "Orange": "orange.jpg",
        "Apple": "apple.jpg",
        "Muskmelon": "muskmelon.jpg",
        "Watermelon": "watermelon.jpg",
        "Grapes": "grapes.jpg",
        "Mango": "mango.jpg",
        "Banana": "banana.jpg",
        "Pomegranate": "pomegranate.jpg",
        "Lentil": "lentil.jpg",
        "Blackgram": "blackgram.jpg",
        "Mungbean": "mungbean.jpg",
        "Mothbeans": "mothbeans.jpg",
        "Pigeonpeas": "pigeonpeas.jpg",
        "Kidneybeans": "kidneybeans.jpg",
        "Chickpea": "chickpea.jpg",
        "Coffee": "coffee.jpg"
    }

    # Map prediction to crop name
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated right there."
        image_file = crop_images.get(crop, "default.jpg")  # Use default image if not found
    else:
        result = "Sorry, we could not determine the best crop with the provided data."
        image_file = "default.jpg"

    # Render result back to the HTML page
    return render_template('index.html', result=result, image_file=image_file)

# -------------------------------
# Run Flask app (Debug Mode)
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)

# -------------------------------
# (Optional) Example: Saving a scaler
# -------------------------------
# If you ever need to save a new scaler after fitting:
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(ms, f)
