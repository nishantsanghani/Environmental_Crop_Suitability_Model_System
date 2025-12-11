🌱 Environmental Crop Suitability Model System
📌 Project Overview

This project is a Machine Learning-based Crop Recommendation System that suggests the most suitable crop for cultivation based on soil and environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall.

## The system includes:
- A trained ML model built using agricultural datasets.
- A Flask web application for user-friendly interaction.
- Scalers (StandardScaler & MinMaxScaler) for feature preprocessing.
- Crop images displayed along with predictions for better visualization.

🚀 Features
- Predicts the best crop to cultivate based on input soil/environment conditions.
- Provides a web interface built with Flask.
- Uses machine learning classification model trained on agricultural data.
- Displays crop-specific images with prediction results.

🛠️ Tools & Technologies
- Python (NumPy, Pandas, scikit-learn, Pickle)
- Flask (for web app development)
- HTML, CSS, Bootstrap (for frontend UI)
- Jupyter Notebook (for model training & analysis)

📂 Project Structure
├── app.py                     # Flask web application  
├── Crop Recommendation.ipynb  # Model training and analysis notebook  
├── model.pkl                  # Trained ML model  
├── standscaler.pkl            # StandardScaler for feature scaling  
├── minmaxscaler.pkl           # MinMaxScaler for feature scaling  
├── templates/  
│   └── index.html             # Frontend HTML page  
├── static/  
│   └── images/                # Crop images (rice.jpg, maize.jpg, etc.)  
└── README.md                  # Project documentation  

⚙️ Installation & Setup

1. Clone the repository:
git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation

2. Install dependencies:
pip install -r requirements.txt

3. Run the Flask app:
python app.py

4. Open in browser:
http://127.0.0.1:5000/

🖥️ Usage
- Enter values for N, P, K, Temperature, Humidity, pH, and Rainfall in the form.
- Click Predict.
- The app will recommend the best crop and display a relevant image.

📊 Example Prediction
# Input:
N = 90, P = 42, K = 43, Temperature = 26°C, Humidity = 80%, pH = 6.5, Rainfall = 200 mm

# Output:
Rice is the best crop to be cultivated right there.

📌 Future Enhancements
- Add real-time weather API integration.
- Deploy on Heroku / AWS / Azure.
- Build a mobile app interface.

Live URL: https://crop-recommendation-system-kle8.onrender.com/
