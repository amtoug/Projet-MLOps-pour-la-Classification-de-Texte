import sys
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2\src\models")
# sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
# from base_model import BaseModel
# # from base_model import BaseModel
import sys
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")

from src.models.base_model import BaseModel

# print(BaseModel)


class LogisticModel(BaseModel):
    def __init__(self,  max_iter: int = 1000):
        self.model = LogisticRegression(max_iter=max_iter)

    def entrainer(self, X: Union[pd.Series, np.ndarray], y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predire(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def evaluer(self, X:Union[pd.Series, np.ndarray], y: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predire(X)
        return classification_report(y, y_pred, output_dict=True)