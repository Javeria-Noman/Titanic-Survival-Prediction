**Titanic Survival Predictor**

A machine learning project that predicts whether a Titanic passenger would have survived, based on details like gender, age, ticket class and fare. The trained model is served through a Flask web app with an interactive, ship-themed interface.

**Overview**

Task: Binary classification (1 = survived, 0 = did not survive)
Dataset: Titanic - Machine Learning from Disaster (Kaggle), 891 passengers in train.csv
Model: Random Forest (tuned), compared against Logistic Regression
App: Flask backend + custom HTML/CSS/JS frontend

**Workflow**

Explore the data: survival rate by gender and class, missing values
Clean and preprocess
Dropped PassengerId, Name, Ticket, Cabin (identifiers or mostly empty)
Filled missing Age with the median and missing Embarked with the mode
Encoded Sex (male = 0, female = 1) and one-hot encoded Embarked
Split: 80% train / 20% test, stratified
Train: Logistic Regression (baseline) and Random Forest
Evaluate: accuracy, confusion matrix, classification report, 5-fold cross-validation
Tune: limited tree depth to reduce overfitting
Deploy: saved the model with joblib and built a Flask web app

**Results**

Model	Train accuracy	Test accuracy	5-fold CV
Random Forest (default)	98.2%	81.0%	80.9%
Random Forest (tuned: max_depth=5, min_samples_leaf=3)	86.2%	79.9%	81.6%

The default Random Forest overfit (17% gap between train and test). Restricting tree depth and leaf size reduced the gap to about 6% and gave the best cross-validation score, so the tuned model is used in the app.

Key insight: gender and ticket class are the strongest predictors, which matches the historical "women and children first" evacuation rule.

**Project Structure**

├── Titanic Survival.ipynb   # data cleaning, training, evaluation

├── train.csv                # Kaggle dataset

├── app.py                   # Flask backend

├── titanic_model.pkl        # saved model + column order

├── requirements.txt

└── templates/

    └── index.html           # frontend

Run Locally
bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000 in your browser.

To retrain and re-save the model, run the notebook and execute:

python
import joblib
joblib.dump({"model": rf2, "columns": list(X.columns)}, "titanic_model.pkl")

The model was trained with the scikit-learn version listed in requirements.txt. Using a different version may cause a pickle warning or error.

Tech Stack

Python, pandas, NumPy, scikit-learn, Flask, joblib, HTML/CSS/JavaScript

Future Improvements
Feature engineering: extract Title (Mr, Mrs, Miss) from names and add FamilySize
Hyperparameter search with GridSearchCV
Try Gradient Boosting or XGBoost
Author

Your name · your LinkedIn / GitHub link
