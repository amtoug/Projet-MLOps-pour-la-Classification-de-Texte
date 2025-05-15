from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
from src.models.base_model import BaseModel

class GradientBoostingModel(BaseModel):

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate, max_depth=max_depth)

    def entrainer(self,X: Union[pd.Series, np.ndarray], y: pd.Series) -> None:
        self.model.fit(X, y)

    def evaluer(self, X: Union[pd.Series, np.ndarray], y: pd.Series) -> Dict[str, Any]:
        y_pred = self.predire(X)
        return classification_report(y, y_pred, output_dict=True)

    def predire(self, X: pd.Series) -> np.ndarray:
        return self.model.predict(X)
