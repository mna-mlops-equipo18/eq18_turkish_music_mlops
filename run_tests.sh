#!/bin/bash

# Script para ejecutar pruebas del proyecto Turkish Music Emotion MLOps
# Uso: ./run_tests.sh [opción]

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Turkish Music Emotion - Test Runner ===${NC}\n"

# Función para mostrar uso
show_usage() {
    echo "Uso: ./run_tests.sh [opción]"
    echo ""
    echo "Opciones:"
    echo "  all           - Ejecutar todas las pruebas (default)"
    echo "  unit          - Solo pruebas unitarias"
    echo "  integration   - Solo pruebas de integración"
    echo "  coverage      - Ejecutar con reporte de cobertura"
    echo "  prepare       - Solo pruebas de prepare.py"
    echo "  train         - Solo pruebas de train.py"
    echo "  evaluate      - Solo pruebas de evaluate.py"
    echo "  transformers  - Solo pruebas de transformers.py"
    echo "  mlflow        - Solo pruebas de mlflow.py"
    echo "  fast          - Pruebas rápidas (excluye lentas)"
    echo "  help          - Mostrar esta ayuda"
    echo ""
}

# Verificar que pytest está instalado
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest no está instalado${NC}"
    echo "Instalar con: pip install pytest pytest-cov"
    exit 1
fi

# Procesar argumentos
case "${1:-all}" in
    all)
        echo -e "${YELLOW}Ejecutando todas las pruebas...${NC}"
        pytest -v
        ;;
    unit)
        echo -e "${YELLOW}Ejecutando pruebas unitarias...${NC}"
        pytest tests/test_prepare.py tests/test_train.py tests/test_evaluate.py \
               tests/test_transformers.py tests/test_mlflow_utils.py -v
        ;;
    integration)
        echo -e "${YELLOW}Ejecutando pruebas de integración...${NC}"
        pytest tests/test_integration_pipeline.py -v
        ;;
    coverage)
        echo -e "${YELLOW}Ejecutando con cobertura...${NC}"
        pytest --cov=eq18_turkish_music_mlops --cov-report=html --cov-report=term
        echo -e "${GREEN}Reporte HTML generado en: htmlcov/index.html${NC}"
        ;;
    prepare)
        echo -e "${YELLOW}Ejecutando pruebas de prepare.py...${NC}"
        pytest tests/test_prepare.py -v
        ;;
    train)
        echo -e "${YELLOW}Ejecutando pruebas de train.py...${NC}"
        pytest tests/test_train.py -v
        ;;
    evaluate)
        echo -e "${YELLOW}Ejecutando pruebas de evaluate.py...${NC}"
        pytest tests/test_evaluate.py -v
        ;;
    transformers)
        echo -e "${YELLOW}Ejecutando pruebas de transformers.py...${NC}"
        pytest tests/test_transformers.py -v
        ;;
    mlflow)
        echo -e "${YELLOW}Ejecutando pruebas de mlflow.py...${NC}"
        pytest tests/test_mlflow_utils.py -v
        ;;
    fast)
        echo -e "${YELLOW}Ejecutando pruebas rápidas...${NC}"
        pytest -m "not slow" -v
        ;;
    help)
        show_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Error: Opción inválida '${1}'${NC}\n"
        show_usage
        exit 1
        ;;
esac

# Mostrar resultado
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Pruebas completadas exitosamente${NC}"
else
    echo -e "\n${RED}✗ Algunas pruebas fallaron${NC}"
    exit 1
fi
