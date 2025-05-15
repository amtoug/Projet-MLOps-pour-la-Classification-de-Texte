from datasets import load_dataset
from classPreTraitement import PreTraitement
from typing import Generator
import pandas as pd

# Générateur de données par lots
def generer_donnees(dataset: pd.DataFrame, taille_lot: int = 1000) -> Generator:
    for debut in range(0, len(dataset), taille_lot):
        fin = min(debut + taille_lot, len(dataset))
        yield dataset.iloc[debut:fin]



def generer_et_traiter_donnees(split: str, filename: str)->None:
    dataset = load_dataset("imdb", split=split)
    df = pd.DataFrame(dataset)

    # Écriture initiale avec l'en-tête
    df_empty = pd.DataFrame(columns=["label", "text"])
    df_empty.to_csv(filename, mode='w', header=True, index=False)

    for lot in generer_donnees(df, taille_lot=1000):
        traitement = PreTraitement(lot)
        traitement.SupprimerValMonquent()
        lot_cleaned = traitement.dataset["text"].apply(traitement.remove_specials)
        traitement.dataset["text"] = lot_cleaned
        traitement.dataset.to_csv(filename, mode='a', header=False, index=False)


# Traitement complet du dataset par lots de 1000
generer_et_traiter_donnees("train", "../../data/processed/resultat_train.csv")
generer_et_traiter_donnees("test", "../../data/processed/resultat_test.csv")
