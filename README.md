# Turkish Music Emotion

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
Clasificación de Emociones en Música Turca Usando Machine Learning

## Equipo 18
- Ali Mateo Campos Martínez / A01796071 / a1licampos
- Mario Fonseca Martínez / A01795228 / mariofmtz15
- Miguel Ángel Hernández Núñez / A01795751 / mickyhn
- Jonatan Israel Meza Mendoza / A01275322 / Jonatana01275322
- Eder Mauricio Castillo Galindo / A01795453 / maurocastill

## Resumen del proyecto

**Nombre:** Clasificación de Emociones en Música Turca
**Propósito:** pendiente*.
Estructura basada en **Cookiecutter Data Science**. Control de código con **Git**, control de datos con **DVC**, y almacenamiento remoto de datos en **Azure Blob Storage**.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         eq18_turkish_music_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── eq18_turkish_music_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes eq18_turkish_music_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Estructura clave del repo

* `data/raw/` — datasets originales (versionados con DVC).
* `notebooks/` — notebooks para EDA y modelado.
* `eq18_turkish_music_mlops/` — código fuente del proyecto.
* `.dvc/` — metadata de DVC (versionada).
* `dvc.yaml` / `dvc.lock` — (si aplican) pipelines DVC.
* `requirements.txt` — dependencias Python.
* `README.md` — este archivo.


## 1. Requisitos

| Requisito | Descripción |
|------------|-------------|
| **Python** | Recomendado: 3.11 (verificado en `pyproject.toml`) |
| **Git** | Para clonar el repositorio |
| **Azure Storage Key** | Archivo `azure_key.txt` con la *Account Key* de tu contenedor |
| *(Opcional)* **Servidor MLflow** | VM con ≥4GB RAM, ≥30GB disco, y puerto `5000` abierto |

---

## Paso 1: Clonar el Repositorio

```bash
# Clona el repositorio
git clone <URL_DE_TU_REPOSITORIO>

# Entra a la carpeta
cd eq18_turkish_music_mlops

# Cambiar a la rama development
git checkout development

# 1. Crea el entorno 
python3 -m venv .venv

# 2. Activar el entorno virtual
source .venv/bin/activate

```
---

## Paso 2. Configuración del Entorno 
```bash

# 1. Crea el entorno 
python3 -m venv .venv

# 2. Activar entorno virtual
source .venv/bin/activate

```
---

## Paso 3. Crear Swap

```bash

sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

```
---

## Paso 4. Instalar Dependencias
Recomendamos altamente empezar a instalar lo mas pesado al principio.

```bash

pip install torch
pip install pandas numpy scikit-learn
pip install "dvc[azure]" mlflow xgboost
pip install -r requirements.txt

```
---

## Paso 5. Conectar DVC a Azure
Teniendo en cuenta que ya tienes tu llave (azure_key.txt) en tu maquina.

```bash

# moveremos la llave al root del proyecto
mv /ruta/a/tu/azure_key.txt .
AZURE_KEY_VALUE=$(cat azure_key.txt)
dvc remote modify azure-storage account_key "$AZURE_KEY_VALUE" --local

```
---

## Paso 6. Descargar los datos

```bash

dvc pull

```
---