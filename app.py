from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

bundle = joblib.load("titanic_model.pkl")
model, columns = bundle["model"], bundle["columns"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json()

    # Same encoding as the notebook: Sex male=0/female=1, Embarked one-hot (C is the dropped base)
    row = {c: 0 for c in columns}
    row.update({
        "Pclass": int(d["pclass"]),
        "Sex": 1 if d["sex"] == "female" else 0,
        "Age": float(d["age"]),
        "SibSp": int(d["sibsp"]),
        "Parch": int(d["parch"]),
        "Fare": float(d["fare"]),
    })
    if d["embarked"] in ("Q", "S"):
        row["Embarked_" + d["embarked"]] = 1

    X = pd.DataFrame([row])[columns]
    p = float(model.predict_proba(X)[0][1])
    return jsonify(probability=round(p * 100, 1), survived=p >= 0.5)


if __name__ == "__main__":
    app.run(debug=True)
