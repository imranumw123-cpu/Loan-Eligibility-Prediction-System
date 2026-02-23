from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to the saved model
MODEL_PATH = 'models/loan_rf_model.pkl'

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Success: Model loaded!")
else:
    model = None
    print("❌ Error: Model file not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # VERY IMPORTANT: The order of columns MUST match X_train.columns from your notebook
        # Standard order for this dataset:
        cols_order = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        # Convert dictionary to DataFrame with specific column order
        df_input = pd.DataFrame([data])[cols_order]
        
        # Make prediction
        prediction = model.predict(df_input)
        status = "Approved" if int(prediction[0]) == 1 else "Rejected"
        
        return jsonify({'loan_status': status})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
