from flask import Flask, render_template, request
import pickle
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load trained model
with open("house_model_india.pkl", "rb") as f:
    model = pickle.load(f)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Change if using Atlas
db = client["house_price_db"]
predictions_col = db["predictions"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("City")
        locality = request.form.get("Locality")
        property_type = request.form.get("Property_Type")
        bhk = int(request.form.get("BHK"))
        size = float(request.form.get("Size_in_SqFt"))
        furnished = request.form.get("Furnished_Status")
        parking = request.form.get("Parking_Space")
        year_built = int(request.form.get("Year_Built"))

        # Prepare input data for model
        input_data = pd.DataFrame([{
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Furnished_Status": furnished,
            "Parking_Space": parking,
            "Year_Built": year_built
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        predicted_price = round(prediction, 2)

        # Store prediction in MongoDB
        record = {
            "City": city,
            "Locality": locality,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Furnished_Status": furnished,
            "Parking_Space": parking,
            "Year_Built": year_built,
            "Predicted_Price_Lakhs": predicted_price,
            "Timestamp": datetime.now()
        }
        predictions_col.insert_one(record)

        return render_template("result.html", price=predicted_price)

    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route("/gallery")
def gallery():
    return render_template("gallery.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
