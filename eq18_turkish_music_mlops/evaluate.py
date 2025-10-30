import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def main(model_name):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    processed_dir = Path(params["paths"]["processed_dir"])
    models_dir = Path(params["paths"]["models_dir"])
    reports_dir = Path(params["paths"]["reports_dir"])

    # --- Cargar datos y modelo ---
    print("Cargando datos y modelo...")
    df_test = pd.read_csv(processed_dir / "test.csv")
    model = joblib.load(models_dir / f"model_{model_name}.pkl")
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")

    X_test = df_test.drop(columns=["Class"])
    y_true = df_test["Class"]
    y_true_encoded = label_encoder.transform(y_true)

    # --- Predicciones ---
    print("Generando predicciones...")
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # --- Métricas ---
    acc = accuracy_score(y_true_encoded, y_pred_encoded)
    f1 = f1_score(y_true_encoded, y_pred_encoded, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # --- Guardar reporte ---
    results = {
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(reports_dir / f"evaluate_results_{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluación completada. Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
    print(f"Reporte guardado en reports/evaluate_results_{model_name}.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluación final del modelo")
    parser.add_argument("--model", type=str, choices=["logistic", "randomforest", "xgboost"], required=True)
    args = parser.parse_args()
    main(args.model)