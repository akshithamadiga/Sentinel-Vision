from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
from detect import predict_image
import os

app = Flask(__name__)

os.makedirs("logs", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filename = "uploaded.jpg"
    path = os.path.join("static", filename)

    file.save(path)

    label, confidence = predict_image(path)

    # Logging (optional)
    ip = request.remote_addr
    time = datetime.now()

    log = pd.DataFrame([[ip, time, label]],
                       columns=["IP","Time","Result"])

    log.to_csv("logs/ip_logs.csv", mode="a", header=False, index=False)

    return render_template(
        "result.html",
        result=label,
        confidence=confidence,
        image_path=path
    )

if __name__ == "__main__":
    app.run(debug=True)