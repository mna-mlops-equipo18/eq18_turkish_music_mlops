import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def main():
    # --- Cargar parámetros ---
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_path = params["paths"]["raw_data"]
    processed_dir = Path(params["paths"]["processed_dir"])
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    target_col = params["data"]["target_col"]
    valid_classes = params["data"]["valid_classes"]

    # --- Cargar datos ---
    print("Cargando datos...")
    df = pd.read_csv(raw_path)

    # --- Limpieza estructural ---
    print("Limpieza estructural...")
    df.columns = df.columns.str.strip()

    # Eliminar columnas irrelevantes
    if "mixed_type_col" in df.columns:
        df = df.drop(columns=["mixed_type_col"])

    if target_col in df.columns:
        df[target_col] = df[target_col].str.strip().str.lower()
        df = df[df[target_col].isin(valid_classes)]
    
	# Reemplazar comas por puntos y limpiar objetos numéricos (excepto Class)
    for col in df.select_dtypes(include="object").columns:
        if col != target_col:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(" ", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].round(3)
		
	# Eliminar duplicados
    df = df.drop_duplicates()
		
    # --- División Train/Test ---
    print("Dividiendo train/test...")
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )

    processed_dir.mkdir(exist_ok=True, parents=True)
    train.to_csv(processed_dir / "train.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    print(f"Archivos guardados en {processed_dir}/")

if __name__ == "__main__":
    main()