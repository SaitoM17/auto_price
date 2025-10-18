import pandas as pd
import kagglehub
import os

# Configuración ruta dentro del proyecto
path_personalizada = os.path.join(os.getcwd(), 'data/raw')

# Creación de la carpeta en caso de que no exista
os.makedirs(path_personalizada, exist_ok=True)

# Inidicar que carpeta usar como cache
os.environ['KAGGLEHUB_CACHE'] = path_personalizada

# Acceder al conjunto de datos de Kaggle
path = kagglehub.dataset_download('metawave/vehicle-price-prediction')
print('Conjunto de datos descargado en:', path)

# Revisar el conjunto de datos descargado
csv_path = os.path.join(path, 'vehicle_price_prediction.csv')
df = pd.read_csv(csv_path)
print(df)