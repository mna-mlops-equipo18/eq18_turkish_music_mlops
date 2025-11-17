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
| **Python** | Recomendado: `3.11` (verificado en `pyproject.toml`) |
| **RAM** | **Mínimo 4GB.** El `pip install` fallará con menos. |
| **Disco** | Mínimo 30GB de almacenamiento. |
| **Azure Storage Key** | Archivo `azure_key.txt` con la *Account Key* de tu contenedor. |

---

## Paso 1: Instalar Docker

```bash
# 1. Actualizar el gestor de paquetes
sudo apt-get update

# 2. Instalar Docker
sudo apt-get install -y docker.io

# 3. Añadir tu usuario al grupo de Docker 
sudo usermod -aG docker $USER

exit
```
---

## Paso 2. Configuración del Entorno 
```bash

# 1. Descarga la imagen de dockerhub
docker pull mariofonsecamtz/eq18_turkish_music_mlops:latest

# 2. Corre el contenedor
docker run \
    -d \
    -p 8000:8000 \
    --name mi-api \
    mariofonsecamtz/eq18_turkish_music_mlops:latest
```
---

## Paso 3. Verificación

```bash
docker ps
```
---

## Paso 4. Probar
Abrir navegador web y visitar la página de swagger, que nos ayudará a validar si nuestra api está funcionando.

http://<IP_EXTERNA_SERVIDOR>:8000/docs

---
