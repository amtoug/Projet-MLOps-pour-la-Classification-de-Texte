import sys
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
import src.data.loader as loader
import pandas as pd
from src.models.factory import get_model

def Model(nameModel:str,*args:Any,**kargs:Any):
    model=get_model(nameModel,*args,**kargs)
    generator = loader.generer_donnees_train()
    chunk_1 = next(generator)
    vectorizer = TfidfVectorizer()
    X = chunk_1["text"]
    y = np.array(chunk_1["label"])
    X_vect = vectorizer.fit_transform(X)
    # y_bin = np.where(y <= 1, 0, 1)
    model.entrainer(X_vect, y) 
    for chunk in generator:
        X=chunk["text"]
        y=chunk["label"]
        X_vect = vectorizer.transform(X)
        y = np.array(y)
        model.entrainer(X_vect,y)
    return model.evaluer(X_vect,y),model,vectorizer

# Model("logistic")
