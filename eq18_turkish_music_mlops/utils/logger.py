import logging
import sys
from pathlib import Path

def setup_logging():
    """
    Configura el logging para todo el proyecto.

    Se encarga de:
    1. Crear el directorio 'logs/' si no existe.
    2. Configurar el formato del log.
    3. Añadir handlers para escribir a un archivo (pipeline.log)
       y a la consola (stdout).
    """
    
    # --- 1. Definir y crear el directorio de logs ---
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(exist_ok=True)

    # --- 2. Configuración básica ---
    # (Solo se configura una vez, la primera vez que se llama)
    logging.basicConfig(
        level=logging.INFO,
        
        # Formato del log
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        
        # Manejadores (a dónde enviar los logs)
        handlers=[
            # Escribir todos los logs en un solo archivo
            logging.FileHandler(LOGS_DIR / "pipeline.log"),
            
            # Mostrar los logs en la consola
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Configuración de logging cargada.")