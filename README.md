# API Prêt Bancaire - Backend

API FastAPI pour la prédiction d'approbation de prêts bancaires utilisant un modèle XGBoost.

## Installation

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Fichier modèle requis

Assurez-vous d'avoir le fichier `loan_approval_model.joblib` dans le même répertoire que `model.py`.

## Lancement

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'API
uvicorn model:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

## Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Information sur l'API |
| `/health` | GET | État de santé de l'API |
| `/predict` | POST | Prédiction avec analyse détaillée |
| `/predict/simple` | POST | Prédiction simplifiée |
| `/features/importance` | GET | Importance des features |

## Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_gender": "male",
    "person_education": "Master",
    "person_income": 75000,
    "person_emp_exp": 10,
    "person_home_ownership": "OWN",
    "loan_amnt": 15000,
    "loan_intent": "PERSONAL",
    "loan_int_rate": 7.5,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 8,
    "credit_score": 750,
    "previous_loan_defaults_on_file": "No"
  }'
```

## Technologies

- FastAPI
- XGBoost
- Scikit-learn
- Pydantic
