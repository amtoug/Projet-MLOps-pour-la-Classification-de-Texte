from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
from src.models.base_model import BaseModel


class MLPModel(BaseModel):

    def __init__(self, hidden_layer_sizes:Tuple[int, ...]=(100,), activation:str='relu', solver:str='adam', max_iter:int=200):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,max_iter=max_iter)

    def entrainer(self, X: Union[pd.Series, np.ndarray], y: np.ndarray):
        self.model.fit(X, y)

    def evaluer(self, X:Union[pd.Series, np.ndarray], y:np.ndarray):
        y_pred = self.predire(X)
        return classification_report(y, y_pred, output_dict=True)

    def predire(self, X:Union[pd.Series, np.ndarray]):
        return self.model.predict(X)
