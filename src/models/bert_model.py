from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from .base_model import StrategieModele
from typing import Any, Dict
import torch

class ModeleBERT(StrategieModele):
    def __init__(self) -> None:
        # Chargement du modèle BERT préentraîné pour la classification à 5 classes
        self.modele = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.trainer = None

    def entrainer(self, X_train: Any, y_train: Any) -> None:
        from datasets import Dataset

        # Conversion des données en dataset compatible avec Hugging Face
        train_dataset = Dataset.from_dict({"text": X_train, "labels": y_train})
        # Tokenisation des textes
        train_dataset = train_dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True, padding="max_length"),
            batched=True
        )

        # Fonction de métrique pour Trainer
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=1)
            return {"accuracy": accuracy_score(labels, preds)}

        # Configuration et initialisation du Trainer
        self.trainer = Trainer(
            model=self.modele,
            args=TrainingArguments(
                output_dir="./bert_output",
                per_device_train_batch_size=8,
                num_train_epochs=2,
                logging_dir="./logs",
                logging_steps=10
            ),
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )

        # Entraînement du modèle
        self.trainer.train()

    def evaluer(self, X_test: Any, y_test: Any) -> float:
        from datasets import Dataset

        # Préparation du dataset de test
        test_dataset = Dataset.from_dict({"text": X_test, "labels": y_test})
        test_dataset = test_dataset.map(
            lambda e: self.tokenizer(e["text"], truncation=True, padding="max_length"),
            batched=True
        )

        # Évaluation du modèle avec le Trainer
        evaluation_result = self.trainer.evaluate(eval_dataset=test_dataset)

        # Retour de la précision (accuracy) mesurée
        return evaluation_result.get("eval_accuracy", 0.0)

    def predire(self, X: Any) -> Any:
        # Préparation des entrées
        inputs = self.tokenizer(X, return_tensors="pt", padding=True, truncation=True)
        # Prédiction avec le modèle
        outputs = self.modele(**inputs)
        return torch.argmax(outputs.logits, dim=1).numpy()
