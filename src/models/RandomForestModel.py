from typing import Dict, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
from src.models.base_model import BaseModel
import pandas as pd
from collections import Counter
# from typing import rep

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators:int=100, max_depth:int=None, random_state:int=42)->None:
        self.model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state)
    
    def entrainer(self, X:pd.DataFrame, y:pd.Series)->None:
        self.model.fit(X, y)

    # def evaluer(self, X:pd.DataFrame, y:pd.Series)->Dict:
    def evaluer(self, X:Union[pd.Series, np.ndarray], y:pd.Series)->Dict:
        y_pred = self.predire(X)
        return classification_report(y, y_pred, output_dict=True)

    def predire(self, X:Union[pd.Series, np.ndarray])->int:
        return self.model.predict(X)
















# from .base_model import BaseModel
# from sklearn.metrics import accuracy_score
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense

# class LSTMModel(BaseModel):
#     def __init__(self, vocab_size=10000, embedding_dim=128, lstm_units=64, num_classes=2):
#         self.model = Sequential([
#             Embedding(input_dim=vocab_size, output_dim=embedding_dim),
#             LSTM(lstm_units),
#             Dense(num_classes, activation="softmax" if num_classes > 1 else "sigmoid")
#         ])
#         self.model.compile(
#             loss="sparse_categorical_crossentropy" if num_classes > 1 else "binary_crossentropy",
#             optimizer="adam",
#             metrics=["accuracy"]
#         )

#     def train(self, X, y, epochs=5, batch_size=32):
#         self.model.fit(np.array(X), np.array(y), epochs=epochs, batch_size=batch_size, verbose=0)

#     def predict(self, X):
#         preds = self.model.predict(np.array(X))
#         if preds.shape[1] == 1:a
#             return (preds > 0.5).astype("int32").flatten()
#         return np.argmax(preds, axis=1)

#     def evaluate(self, X, y):
#         preds = self.predict(X)
#         return accuracy_score(y, preds)

#     def save(self, path="lstm_model.keras"):
#         self.model.save(path)

#     def load(self, path="lstm_model.keras"):
#         self.model = tf.keras.models.load_model(path)