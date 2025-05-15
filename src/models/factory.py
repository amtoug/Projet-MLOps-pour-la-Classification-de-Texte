import sys
from typing import Any
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")

from src.models.RandomForestModel import RandomForestModel
from src.models.logistic_model import LogisticModel
from src.models.GradientBoostingModel import GradientBoostingModel 
from src.models.MLPClassifier import MLPModel 

def get_model(model_name: str,*args:Any,**kargs:Any):
    if model_name == "GradientBoostingModel":
        return GradientBoostingModel(*args)
    elif model_name == "RandomForestModel":
        return RandomForestModel(*args)
    elif model_name == "logistic":
        return LogisticModel(*args)
    elif model_name=="MLPClassifier":
        return MLPModel(*args,**kargs)
    else:
        raise ValueError("Modèle inconnu")
