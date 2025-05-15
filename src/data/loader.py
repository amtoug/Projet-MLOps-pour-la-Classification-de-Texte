import pandas as pd
from typing import Generator
# import sys
# sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexte\src\utils")
# # print(sys.path)
# import logger


# logger.logging()    
def generer_donnees_train(taille_lot: int = 1000) -> Generator:
    return pd.read_csv("../../data/processed/resultat_train.csv",chunksize=taille_lot)

def generer_donnees_test(taille_lot: int = 1000) -> Generator:
    return pd.read_csv("../../data/processed/resultat_test.csv",chunksize=taille_lot)


# for chunk in generer_donnees_Enter():
#     print(chunk.head())  # Affiche les 5 premières lignes du premier chunk
#     break 