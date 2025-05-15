from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
import sys
import os
sys.path.append(r"C:\Users\Abdessamad\Desktop\MLOpsClassificationTexteV2")
import src.training.train as train
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import bcrypt
from typing import Any, Callable, List,Dict, Optional, Tuple, Union
from prometheus_client import start_http_server, Gauge,generate_latest
from fastapi.responses import PlainTextResponse
import time
from src.models.factory import get_model
from src.utils.logger import logging
from src.utils.decorators import log_execution_time
from validationDataPydantic import DataBatch
# from data.preprocessing import PreTraitement
import src.data.loader as loader
import numpy as np
from Drift import detect_drift
from sklearn.metrics import accuracy_score
oauth2 = OAuth2PasswordBearer(tokenUrl="connexion")
app = FastAPI()
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
MODEL_DIR = "joblib/"
os.makedirs(MODEL_DIR, exist_ok=True)
def save_model(model: Any, name: str) -> None:
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

@log_execution_time
def load_model(name: str) -> Optional[Any]:
    print(name)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path) if os.path.exists(path) else None


utilisateurs: Dict[str, Dict[str, object]] = {
    "amtoug": {"motdepasse": bcrypt.hashpw(b"12345", bcrypt.gensalt()), "role": "admin"},
    "Abdessamad": {"motdepasse": bcrypt.hashpw(b"12345", bcrypt.gensalt()), "role": "utilisateur"},
}
sessions: Dict[str, dict] = {}

@app.middleware("http")
async def gestion_session(request: Request, call_next) -> Response:
    session_id = request.cookies.get("session_id")
    request.state.session = sessions.get(session_id) if session_id else None
    response = await call_next(request)
    return response

def creer_session(nom_utilisateur: str, response: Response) -> str:
    session_id = str(hash(f"{nom_utilisateur}{datetime.now().timestamp()}"))
    sessions[session_id] = {
        "nom": nom_utilisateur,
        "role": utilisateurs[nom_utilisateur]["role"],
        "expire": datetime.now() + timedelta(minutes=30)
    }
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return session_id

def utilisateur_actuel(request: Request) -> Dict[str, Any]:
    if not request.state.session:
        raise HTTPException(status_code=401, detail="Non authentifié")
    return request.state.session

def role_requis(role_voulu: str) -> Callable[..., Dict[str, Any]]:
    def verificateur(utilisateur: dict = Depends(utilisateur_actuel)):
        if utilisateur["role"] != role_voulu:
            raise HTTPException(status_code=403, detail="Accès interdit")
        return utilisateur
    return verificateur

@app.post("/connexion")
def connexion(request: Request, response: Response, formulaire: OAuth2PasswordRequestForm = Depends()) -> Dict[str, str]:
    utilisateur = utilisateurs.get(formulaire.username)
    if not utilisateur or not bcrypt.checkpw(formulaire.password.encode(), utilisateur["motdepasse"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    logging("warning", f"{formulaire.username} avec role:{utilisateur['role']} est connecté")
    creer_session(formulaire.username, response)
    return {"message": "Connexion réussie"}

@app.post("/deconnexion")
def deconnexion(request: Request, response: Response) -> Dict[str, str]:
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    response.delete_cookie("session_id")
    return {"message": "Déconnecté"}

commentaires: List[str] = []

@app.post("/commentaire")
def mes_donnees(commentaire: str, utilisateur: dict = Depends(utilisateur_actuel))->Union[Dict[str, str], Dict[str, str]]:
    if not commentaire.strip():
        return {"error": "Le commentaire ne peut pas être vide"}
    
    # Stockage du commentaire
    commentaires.append(commentaire)
    return {"message": "Votre commentaire a été ajouté avec succès!", "commentaire": commentaire}

@log_execution_time
@app.get("/commentaires")
def afficher_commentaires()-> Dict[str, Union[str, List[str]]]:
    if not commentaires:
        return {"message": "Aucun commentaire trouvé."}
    return {"commentaires": commentaires}

@log_execution_time
def analyser_commentaires(model: object, vectorizer: object) -> Union[Dict[str, str], List[str]]:
    vect_data = vectorizer.transform(commentaires)
    predictions = model.predire(vect_data)

    results = ["Positive" if p == 1 else "Negative" for p in predictions]
    if not commentaires:
        return {"message": "Aucun commentaire trouvé."}
    return results


@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics()->str:
    return generate_latest()
@log_execution_time
def train_logistic()-> Tuple[Any, Any, float]:
    rapport_train, model_logistic, vectorizer = train.Model("logistic", 46)
    accuracy:float = rapport_train["accuracy"]
    return model_logistic, vectorizer,accuracy


@app.get("/logistic")
async def logistic(utilisateur: dict = Depends(role_requis("admin"))) -> Union[Dict[str, str], List[str]]:
    return analyser_commentaires(model_logistic,vectorizer)
@log_execution_time
def train_MLPClassifier()-> Tuple[Any, Any, float]:
    rapport_train, model_MLPClassifier, _ = train.Model("MLPClassifier", 25)
    accuracy:float = rapport_train["accuracy"]
    return model_MLPClassifier, vectorizer,accuracy

@app.get("/MLPClassifier")
async def logistic(utilisateur = Depends(role_requis("admin"))) -> Union[Dict[str, str], List[str]]:
    return analyser_commentaires(model_MLPClassifier,vectorizer)
@log_execution_time
def train_RandomForestModel()-> Tuple[Any, Any, float]:
    rapport_train, model, _ = train.Model("RandomForestModel", 16)
    accuracy:float = rapport_train["accuracy"]
    return model, vectorizer,accuracy
@app.get("/RandomForestModel")
async def random_forest_model(utilisateur: dict = Depends(role_requis("admin")))-> Union[Dict[str, str], List[str]]:
    return analyser_commentaires(model_RandomForestModel,vectorizer)
@log_execution_time
def train_GradientBoostingModel()-> Tuple[Any, Any, float]:
    rapport_train, model, _ = train.Model("GradientBoostingModel", n_estimators=83, learning_rate=0.2, max_depth=16)
    accuracy:float = rapport_train["accuracy"]
    return model, vectorizer,accuracy
@app.get("/GradientBoostingModel")
async def gradient_boosting_model(utilisateur: dict = Depends(role_requis("admin")))-> Union[Dict[str, str], List[str]]:
    return analyser_commentaires(model_GradientBoostingModel,vectorizer)


accuracy_logistic_train = Gauge('accuracy_logistic_train', 'Précision du modèle')
accuracy_MLPClassifier_train = Gauge('accuracy_MLPClassifier_train', 'Précision du modèle')
accuracy_RandomForestModel_train = Gauge('accuracy_RandomForestModel_train', 'Précision du modèle')
accuracy_GradientBoostingModel_train = Gauge('accuracy_GradientBoostingModel_train', 'Précision du modèle')
model_logistic: Any
model_MLPClassifier: Any
model_RandomForestModel: Any
model_GradientBoostingModel: Any
vectorizer: Any

accuracy_logistique: float
accuracy_GradientBoostingModel: float
accuracy_MLPClassifier: float
accuracy_RandomForestModel: float

reference_vectors: Optional[Any] = None
@app.on_event("startup")
async def startup_event()->None:
    start_http_server(8000) 
    global model_logistic, model_MLPClassifier, model_RandomForestModel, model_GradientBoostingModel, vectorizer
    global accuracy_logistique,accuracy_GradientBoostingModel,accuracy_MLPClassifier,accuracy_RandomForestModel
    global reference_vectors
    reference_vectors = load_model("reference_vectors")

# ---------------------------------------------------------------------
    model_logistic = load_model("model_logistic")
    vectorizer = load_model("vectorizer")
    accuracy_logistique = load_model("accuracy_logistique")

    if model_logistic is None or vectorizer is None or accuracy_logistique is None:
        logging("info","[INFO] Entraînement du modèle Logistic en cours...")
        model_logistic, vectorizer, accuracy_logistique = train_logistic()
        save_model(model_logistic, "model_logistic")
        save_model(vectorizer, "vectorizer")
        save_model(accuracy_logistique, "accuracy_logistique")

    accuracy_logistic_train.set(accuracy_logistique)
# ---------------------------------------------------------------------
    model_MLPClassifier = load_model("model_MLPClassifier")
    accuracy_MLPClassifier = load_model("accuracy_MLPClassifier")

    if model_MLPClassifier is None or accuracy_MLPClassifier is None:
        logging("info","[INFO] Entraînement du modèle MLPClassifier en cours...")
        model_MLPClassifier, _, accuracy_MLPClassifier = train_MLPClassifier()
        save_model(model_MLPClassifier, "model_MLPClassifier")
        save_model(accuracy_MLPClassifier, "accuracy_MLPClassifier")
    accuracy_MLPClassifier_train.set(accuracy_MLPClassifier)
# ---------------------------------------------------------------------
    model_RandomForestModel = load_model("model_RandomForestModel")
    accuracy_RandomForestModel = load_model("accuracy_RandomForestModel")

    if model_RandomForestModel is None or accuracy_RandomForestModel is None:
        logging("info","[INFO] Entraînement du modèle RandomForestModel en cours...")
        model_RandomForestModel, _, accuracy_RandomForestModel = train_RandomForestModel()
        save_model(model_RandomForestModel, "model_RandomForestModel")
        save_model(accuracy_RandomForestModel, "accuracy_RandomForestModel")
    accuracy_RandomForestModel_train.set(accuracy_RandomForestModel)
# ---------------------------------------------------------------------
    model_GradientBoostingModel = load_model("model_GradientBoostingModel")
    accuracy_GradientBoostingModel = load_model("accuracy_GradientBoostingModel")

    if model_GradientBoostingModel is None or accuracy_GradientBoostingModel is None:
        logging("info","[INFO] Entraînement du modèle GradientBoostingModel en cours...")
        model_GradientBoostingModel, _, accuracy_GradientBoostingModel = train_GradientBoostingModel()
        save_model(model_GradientBoostingModel, "model_GradientBoostingModel")
        save_model(accuracy_GradientBoostingModel, "accuracy_GradientBoostingModel")
    accuracy_GradientBoostingModel_train.set(accuracy_GradientBoostingModel)

performance_threshold: float = 0.05
def alert_performance_drop(model: str, previous: float, current: float) -> None:
    if previous - current >= performance_threshold:
        logging("info",f"⚠️ Alerte : Baisse significative de la performance ! Ancienne performance : {previous}, Nouvelle performance : {current} de {model}")
        
@app.post("/retrain")
async def retrain(data: DataBatch,utilisateur: dict = Depends(role_requis("admin")))->str:
    global reference_vectors
    texts:list[str] = [item.text for item in data.data]
    labels:list[int] = [item.label for item in data.data]
    X_new = vectorizer.transform(texts).toarray()
    reference_vectors = reference_vectors.toarray()
    
    if detect_drift(reference_vectors, X_new):
        model_logistic.entrainer(X_new, labels)
        y_pred = model_logistic.predire(X_new)
        acc:float = accuracy_score(labels, y_pred)
        alert_performance_drop("model_logistic", accuracy_logistique, acc)
        save_model(model_logistic, "model_logistic")
        accuracy_logistic_train.set(acc)
        save_model(acc, "accuracy_logistique")
        logging("info", "[INFO] Drift détecté. Modèle logistic réentraîné et sauvegardé...")
        model_MLPClassifier.entrainer(X_new, labels)
        y_pred=model_MLPClassifier.predire(X_new)
        acc:float=accuracy_score(labels,y_pred)
        alert_performance_drop("model_MLPClassifier",accuracy_MLPClassifier, acc)
        save_model(model_MLPClassifier, "model_MLPClassifier")
        accuracy_GradientBoostingModel_train.set(acc)
        save_model(acc, "accuracy_MLPClassifier")
        logging("info","[INFO] Drift détecté. Modèle MLPClassifier réentraîné et sauvegardé...")

        model_GradientBoostingModel.entrainer(X_new, labels)
        save_model(model_GradientBoostingModel, "model_GradientBoostingModel")
        y_pred=model_GradientBoostingModel.predire(X_new)
        acc:float=accuracy_score(labels,y_pred)
        alert_performance_drop("model_GradientBoostingModel",accuracy_GradientBoostingModel, acc)
        accuracy_GradientBoostingModel_train.set(acc)
        save_model(acc, "accuracy_GradientBoostingModel")
        logging("info","[INFO] Drift détecté. Modèle GradientBoostingModel réentraîné et sauvegardé...")

        model_RandomForestModel.entrainer(X_new, labels)
        save_model(model_RandomForestModel, "model_RandomForestModel")
        y_pred=model_RandomForestModel.predire(X_new)
        acc=accuracy_score(labels,y_pred)
        alert_performance_drop("model_RandomForestModel",accuracy_RandomForestModel, acc)
        accuracy_RandomForestModel_train.set(acc)
        logging("info","[INFO] Drift détecté. Modèle RandomForestModel réentraîné et sauvegardé...")
        save_model(acc, "accuracy_RandomForestModel")

        logging("info","[INFO] Drift détecté. Modèles réentraîné et sauvegardé...")
        return f"Drift détecté. Modèles réentraîné et sauvegardé."
    
    logging("info","[INFO] Pas de drift détecté. Aucune action effectuée...")
    return "Pas de drift détecté. Aucune action effectuée."
