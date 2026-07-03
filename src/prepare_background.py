"""
Genera y persiste una muestra de background para SHAP TreeExplainer.

Necesaria porque nodes.py usa model_output="probability" con
feature_perturbation="interventional", lo que requiere un dataset de
referencia frente al que calcular las expectativas de intervencion.

Este script no aparecia en la guia de implementacion original: nodes.py
referenciaba "models/background_sample.joblib" sin que nada lo generase.
Ejecutar SIEMPRE despues de train_model.py y ANTES de graph.py.
"""
import joblib

from data_loader import load_german_credit, split_data

N_BACKGROUND = 100  # tamano habitual: 50-200 filas es suficiente y rapido

if __name__ == "__main__":
    model = joblib.load("models/xgb_scoring_pipeline.joblib")
    preprocessor = model.named_steps["preprocessor"]

    df = load_german_credit("data/german.data")
    X_train, _, _, _ = split_data(df)

    n = min(N_BACKGROUND, len(X_train))
    background_raw = X_train.sample(n=n, random_state=42)
    background_transformed = preprocessor.transform(background_raw)

    joblib.dump(background_transformed, "models/background_sample.joblib")
    print(f"Background guardado: shape={background_transformed.shape}")
