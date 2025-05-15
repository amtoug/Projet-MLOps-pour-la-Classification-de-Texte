# Nom du Projet

Exemple : **MLOps Pipeline for Model Training and Deployment**

---

## Description

Ce projet implémente un pipeline complet MLOps pour l’entraînement, l’évaluation et le déploiement de modèles de machine learning.  
Il utilise DVC pour la gestion des données, FastAPI pour l’API, Grafana & Prometheus pour le monitoring, et Docker pour la containerisation.

---
## Structure du projet

- `data/` : données brutes et prétraitées (non versionnées)
- `src/` : code source (modèles, préprocessing, entraînement, API, etc.)
- `monitoring/` : configurations Grafana et Prometheus
- `Dockerfile` et `docker-compose.yml` : containerisation

---

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/amtoug/Projet-MLOps-pour-la-Classification-de-Texte.git
   cd Projet-MLOps-pour-la-Classification-de-Texte
## Créer un environnement virtuel et installer les dépendances :

python -m venv venv
source venv/bin/activate   # Linux/macOS
# ou
venv\Scripts\activate      # Windows

pip install -r requirements.txt

## Installer et configurer DVC, si ce n’est pas déjà fait.

## Utilisation
## Pipeline de données et modèles avec DVC

dvc repro

## Entraînement manuel
uvicorn src.api.main:app --reload

## Lancement du monitoring avec Docker Compose
docker-compose up

# Contact
Email : amtougabdessamad@gmail.com
GitHub : https://github.com/amtoug
