from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import uvicorn
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Prêt Bancaire - XGBoost",
    description="API de prédiction pour l'approbation de prêts bancaires",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Développement local
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        
        # Votre frontend Vercel (PRODUCTION)
        "https://front-app-bancaire.vercel.app",
        
        # URLs de preview Vercel (si vous en avez)
        "https://vercel.com/f-elwavis-projects",
        
        # Tous les sous-domaines Vercel (pattern générique)
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Chargement du modèle
model = None
try:
    model = joblib.load("loan_approval_model.joblib")
    logger.info(f"✅ Modèle XGBoost chargé (n_features={model.n_features_in_})")
except FileNotFoundError:
    logger.error("❌ Fichier loan_approval_model.joblib non trouvé")
    logger.info("📁 Répertoire actuel: utiliser un chemin absolu ou vérifier le fichier")
except Exception as e:
    logger.error(f"❌ Erreur chargement modèle: {e}")
    model = None

# ENCODAGE - Must match the notebook's LabelEncoder mappings exactly
ENCODING = {
    "gender": {"female": 0, "male": 1, "Female": 0, "Male": 1},  # Support both cases
    "education": {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4},
    "home_ownership": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
    "loan_intent": {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5},
    "previous_defaults": {"No": 0, "Yes": 1}
}

# Enums pour validation - Must include all values from training data
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class Education(str, Enum):
    ASSOCIATE = "Associate"
    BACHELOR = "Bachelor"
    DOCTORATE = "Doctorate"
    HIGH_SCHOOL = "High School"
    MASTER = "Master"

class HomeOwnership(str, Enum):
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"
    OWN = "OWN"
    RENT = "RENT"

class LoanIntent(str, Enum):
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"
    EDUCATION = "EDUCATION"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    MEDICAL = "MEDICAL"
    PERSONAL = "PERSONAL"
    VENTURE = "VENTURE"

class PreviousDefaults(str, Enum):
    NO = "No"
    YES = "Yes"

class LoanRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Âge du demandeur (18-100)")
    person_gender: Gender = Field(..., description="Genre du demandeur")
    person_education: Education = Field(..., description="Niveau d'éducation")
    person_income: float = Field(..., ge=0, description="Revenu annuel en €")
    person_emp_exp: int = Field(..., ge=0, description="Expérience professionnelle en années")
    person_home_ownership: HomeOwnership = Field(..., description="Situation de logement")
    loan_amnt: float = Field(..., ge=0, description="Montant du prêt en €")
    loan_intent: LoanIntent = Field(..., description="Objet du prêt")
    loan_int_rate: float = Field(..., ge=0, le=100, description="Taux d'intérêt en %")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Ratio prêt/revenu (0-1)")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Historique de crédit en années")
    credit_score: int = Field(..., ge=300, le=850, description="Score de crédit (300-850)")
    previous_loan_defaults_on_file: PreviousDefaults = Field(..., description="Défauts antérieurs")

def prepare_features(data: LoanRequest) -> np.ndarray:
    """Prépare les features dans l'ordre attendu par le modèle"""
    features = np.array([[
        float(data.person_age),
        float(ENCODING["gender"][data.person_gender.value]),
        float(ENCODING["education"][data.person_education.value]),
        float(data.person_income),
        float(data.person_emp_exp),
        float(ENCODING["home_ownership"][data.person_home_ownership.value]),
        float(data.loan_amnt),
        float(ENCODING["loan_intent"][data.loan_intent.value]),
        float(data.loan_int_rate),
        float(data.loan_percent_income),
        float(data.cb_person_cred_hist_length),
        float(data.credit_score),
        float(ENCODING["previous_defaults"][data.previous_loan_defaults_on_file.value])
    ]], dtype=np.float32)
    return features

def convert_numpy_types(obj: Any) -> Any:
    """Convertit récursivement les types NumPy en types Python natifs"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def get_feature_importance() -> Dict[str, float]:
    """Récupère l'importance des features depuis XGBoost"""
    if model is not None and hasattr(model, 'feature_importances_'):
        feature_names = [
            "person_age", "person_gender", "person_education", "person_income",
            "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "credit_score", "previous_loan_defaults_on_file"
        ]
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        
        # Normalise à 100%
        total = sum(importance_dict.values())
        if total > 0:
            normalized_dict = {k: float((v/total)*100) for k, v in importance_dict.items()}
        else:
            normalized_dict = {k: 0.0 for k in importance_dict.keys()}
        
        return convert_numpy_types(normalized_dict)
    else:
        # Valeurs par défaut raisonnables
        return {
            "loan_percent_income": 25.0,
            "credit_score": 20.0,
            "person_income": 15.0,
            "loan_int_rate": 10.0,
            "person_age": 8.0,
            "cb_person_cred_hist_length": 6.0,
            "person_emp_exp": 5.0,
            "previous_loan_defaults_on_file": 4.0,
            "person_education": 3.0,
            "person_home_ownership": 2.0,
            "loan_amnt": 1.5,
            "loan_intent": 0.3,
            "person_gender": 0.2
        }

def get_feature_display_name(feature_name: str) -> str:
    """Retourne le nom d'affichage pour une feature"""
    display_names = {
        "person_age": "Âge",
        "person_income": "Revenu Annuel", 
        "credit_score": "Score de Crédit",
        "loan_int_rate": "Taux d'Intérêt",
        "loan_percent_income": "Ratio Prêt/Revenu",
        "cb_person_cred_hist_length": "Historique de Crédit",
        "previous_loan_defaults_on_file": "Défauts Antérieurs",
        "person_emp_exp": "Expérience Professionnelle",
        "person_education": "Niveau d'Études",
        "person_home_ownership": "Situation Logement",
        "loan_amnt": "Montant du Prêt",
        "loan_intent": "Objet du Prêt",
        "person_gender": "Genre"
    }
    return display_names.get(feature_name, feature_name)

def format_feature_value(feature_name: str, raw_value: Any) -> str:
    """Formate la valeur d'une feature pour l'affichage"""
    if feature_name == "person_income" or feature_name == "loan_amnt":
        return f"{float(raw_value):,.0f} €"
    elif feature_name == "loan_percent_income":
        return f"{float(raw_value) * 100:.1f}%"
    elif feature_name == "loan_int_rate":
        return f"{float(raw_value):.1f}%"
    elif feature_name == "previous_loan_defaults_on_file":
        return "Non" if raw_value == "No" else "Oui"
    elif feature_name == "person_gender":
        return "Homme" if str(raw_value).lower() == "male" else "Femme"
    elif feature_name == "person_education":
        education_map = {
            "Associate": "DUT/BTS",
            "Bachelor": "Licence",
            "Doctorate": "Doctorat",
            "High School": "Lycée", 
            "Master": "Master"
        }
        return education_map.get(str(raw_value), str(raw_value))
    elif feature_name == "person_home_ownership":
        home_map = {
            "MORTGAGE": "Hypothèque",
            "OTHER": "Autre",
            "OWN": "Propriétaire",
            "RENT": "Locataire"
        }
        return home_map.get(str(raw_value), str(raw_value))
    elif feature_name == "loan_intent":
        intent_map = {
            "DEBTCONSOLIDATION": "Consolidation de dettes",
            "EDUCATION": "Éducation",
            "HOMEIMPROVEMENT": "Amélioration habitat",
            "MEDICAL": "Médical",
            "PERSONAL": "Usage personnel", 
            "VENTURE": "Projet entrepreneurial"
        }
        return intent_map.get(str(raw_value), str(raw_value))
    else:
        return str(raw_value)

def evaluate_feature_status(feature_name: str, value: Any) -> str:
    """Évalue le statut d'une feature (Favorable/Neutre/Risqué)"""
    try:
        numeric_value = float(value)
        
        if feature_name == "loan_percent_income":
            return "Favorable" if numeric_value <= 0.25 else "Neutre" if numeric_value <= 0.35 else "Risqué"
        elif feature_name == "credit_score":
            return "Favorable" if numeric_value >= 750 else "Neutre" if numeric_value >= 650 else "Risqué"
        elif feature_name == "person_income":
            return "Favorable" if numeric_value >= 80000 else "Neutre" if numeric_value >= 40000 else "Risqué"
        elif feature_name == "loan_int_rate":
            return "Favorable" if numeric_value <= 8 else "Neutre" if numeric_value <= 12 else "Risqué"
        elif feature_name == "person_age":
            return "Favorable" if 25 <= numeric_value <= 55 else "Neutre" if numeric_value > 55 else "Risqué"
        elif feature_name == "person_emp_exp":
            return "Favorable" if numeric_value >= 5 else "Neutre" if numeric_value >= 2 else "Risqué"
        elif feature_name == "person_education":
            # Encoding: Associate:0, Bachelor:1, Doctorate:2, High School:3, Master:4
            # Higher education (Doctorate=2, Master=4) is favorable
            return "Favorable" if numeric_value in [2, 4] else "Neutre"
        elif feature_name == "person_home_ownership":
            # Encoding: MORTGAGE:0, OTHER:1, OWN:2, RENT:3
            # OWN (2) is favorable, MORTGAGE (0) is neutral, RENT/OTHER less favorable
            return "Favorable" if numeric_value == 2 else "Neutre" if numeric_value == 0 else "Risqué"
        else:
            return "Neutre"
    except (ValueError, TypeError):
        # Pour les features non-numériques
        if feature_name == "previous_loan_defaults_on_file":
            return "Favorable" if str(value) == "No" else "Risqué"
        else:
            return "Neutre"

def get_encoded_value(feature_name: str, raw_value: Any) -> Any:
    """Retourne la valeur encodée pour l'évaluation"""
    if feature_name == "person_gender":
        # male:1, female:0 (case-insensitive)
        return 1 if str(raw_value).lower() == "male" else 0
    elif feature_name == "person_education":
        return ENCODING["education"].get(str(raw_value), 1)
    elif feature_name == "person_home_ownership":
        return ENCODING["home_ownership"].get(str(raw_value), 3)  # Default to RENT
    elif feature_name == "loan_intent":
        return ENCODING["loan_intent"].get(str(raw_value), 4)  # Default to PERSONAL
    elif feature_name == "previous_loan_defaults_on_file":
        return ENCODING["previous_defaults"].get(str(raw_value), 0)
    else:
        return raw_value

@app.get("/")
def read_root():
    """Endpoint racine"""
    return {
        "name": "API Prêt Bancaire",
        "version": "1.0.0",
        "description": "API de prédiction pour l'approbation de prêts bancaires",
        "endpoints": {
            "/health": "Vérifier l'état de l'API",
            "/predict": "Faire une prédiction de prêt",
            "/predict/simple": "Version simplifiée de prédiction"
        }
    }

@app.get("/health")
def health_check():
    """Vérifier l'état de l'API et du modèle"""
    status = {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "XGBClassifier" if model else "None",
        "features_count": model.n_features_in_ if model and hasattr(model, 'n_features_in_') else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if model and hasattr(model, 'feature_importances_'):
        status["feature_importance_available"] = True
        status["total_importance"] = float(sum(model.feature_importances_))
    else:
        status["feature_importance_available"] = False
    
    return status

@app.post("/predict")
def predict(data: LoanRequest):
    """Endpoint principal de prédiction avec analyse détaillée"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non disponible. Veuillez vérifier que le fichier loan_approval_model.joblib existe."
        )
    
    try:
        # Préparation des features
        X = prepare_features(data)
        logger.info(f"🔢 Features shape: {X.shape}")
        
        # Prédiction
        prediction_array = model.predict(X)
        prediction = int(prediction_array[0])
        
        # Probabilité
        if hasattr(model, 'predict_proba'):
            probability_array = model.predict_proba(X)[0][1]
            probability = float(probability_array)
        else:
            probability = 0.5
        
        logger.info(f"🎯 Prédiction: {prediction}, Probabilité: {probability:.2%}")
        
        # Récupère l'importance des features
        feature_importance = get_feature_importance()
        
        # Prépare les données brutes
        raw_data = data.dict()
        
        # Crée les facteurs d'analyse
        analysis_factors = []
        
        # Prend les 6 features les plus importantes
        important_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:6]
        
        for feature_name, importance in important_features:
            if feature_name in raw_data:
                raw_value = raw_data[feature_name]
                display_name = get_feature_display_name(feature_name)
                formatted_value = format_feature_value(feature_name, raw_value)
                encoded_value = get_encoded_value(feature_name, raw_value)
                status = evaluate_feature_status(feature_name, encoded_value)
                
                analysis_factors.append({
                    "name": display_name,
                    "value": formatted_value,
                    "status": status,
                    "importance": float(round(importance, 1))
                })
        
        # Génère la raison basée sur l'analyse
        positive_factors = [f for f in analysis_factors if f["status"] == "Favorable"]
        negative_factors = [f for f in analysis_factors if f["status"] == "Risqué"]
        
        if prediction == 1:
            reason = f"Prêt accordé avec une probabilité de {probability:.1%}. "
            if positive_factors:
                reason += f"Facteurs favorables: {', '.join([f['name'] for f in positive_factors[:2]])}. "
        else:
            reason = f"Prêt refusé (probabilité d'acceptation: {probability:.1%}). "
            if negative_factors:
                reason += f"Points à améliorer: {', '.join([f['name'] for f in negative_factors[:2]])}. "
            elif positive_factors:
                reason += f"Points positifs: {', '.join([f['name'] for f in positive_factors[:2]])}, mais insuffisants. "
        
        # Override pour profils exceptionnels
        final_approved = bool(prediction)
        if (prediction == 0 and probability < 0.3 and 
            len(positive_factors) >= 4 and len(negative_factors) == 0):
            final_approved = True
            reason = f"PRÊT ACCEPTÉ (profil exceptionnel). Points forts: {', '.join([f['name'] for f in positive_factors[:3]])}."
        
        # Réponse complète avec conversion des types
        response_data = {
            "approved": final_approved,
            "prediction": prediction,
            "probability": probability,
            "reason": reason,
            "analysis_factors": analysis_factors,
            "feature_importance": {k: float(round(v, 2)) for k, v in feature_importance.items()},
            "model_info": {
                "type": "XGBClassifier",
                "confidence": probability,
                "threshold": 0.5
            }
        }
        
        # Conversion finale pour s'assurer que tout est sérialisable
        return convert_numpy_types(response_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.post("/predict/simple")
def predict_simple(data: LoanRequest):
    """Version simplifiée pour compatibilité"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")
            
        X = prepare_features(data)
        prediction_array = model.predict(X)
        prediction = int(prediction_array[0])
        
        if hasattr(model, 'predict_proba'):
            probability_array = model.predict_proba(X)[0][1]
            probability = float(probability_array)
        else:
            probability = None
        
        return {
            "approved": bool(prediction),
            "prediction": prediction,
            "probability": probability,
            "reason": "Prêt accordé" if prediction == 1 else "Prêt refusé"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/importance")
def get_features_importance():
    """Obtenir l'importance des features du modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    feature_importance = get_feature_importance()
    return {"feature_importance": feature_importance}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Démarrage de l'API sur le port {port}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
