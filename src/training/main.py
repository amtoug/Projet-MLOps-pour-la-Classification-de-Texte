from fastapi import FastAPI,UploadFile
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


     
# Experiment 1: Train Logistic Regression Classifier

log_reg = LogisticRegression(C=3, solver='liblinear')
log_reg.fit(X_train, y_train)
app = FastAPI()

@app.get("/")
def hello(msg:str):
    return {f"message": "FastAPI est opérationnel !{Msg}"}

