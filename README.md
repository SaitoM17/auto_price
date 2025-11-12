# 📊 Automotive Price
# Análisis del Precio de Autómiviles

En este proyecto se presetan un análisis exploratorio de datos y la implementación de algoritmos de aprendizaje automático al conjunto de datos [Automotive Price Prediction Dataset](https://www.kaggle.com/datasets/metawave/vehicle-price-prediction) de [Atul Kumar Choudhary](https://www.kaggle.com/metawave) con el fin de poner en practica las habilidades en el aprendizaje automático.


---

## 📚 Tabla de Contenidos

- [🎯 Propósito](#-propósito)
- [📦 Conjunto de Datos](#-conjunto-de-datos)
- [🧪 Desarrollo del Proyecto](#-desarrollo-del-proyecto)
- [💡 Insights Claves](#-insights-claves)
- [🛠️ Tecnologías](#️-tecnologías)
- [⚙️ Instalación](#️-instalación)
- [👤 Autor](#-autor)
- [📝 Licencia](#-licencia)

---

## 🎯 Propósito

El propósito de este proyecto es explorar cómo variables como marca, año, kilometraje y potencia afectan el valor de un vehículo. Esto como un caso de estudio educativo y profesional, integrando las etapas clave del ciclo de análisi de datos: recolección, limpieza, análisis, modelado y visualización.

---

## 📦 Conjunto de Datos

El conjunto de datos utilizado contiene las siguientes columnas:

- ``make:`` El fabricante o marca del vehículo (por ejemplo, Ford, Toyota).
- ``model:`` El modelo específico del vehículo (por ejemplo, F-150, Camry).
- ``year:`` El año en que se fabricó el vehículo.
- ``mileage:`` La distancia total que ha recorrido el vehículo, expresada en millas.
- ``engine_hp:`` La potencia del motor del vehículo, en caballos de fuerza (horsepower).
- ``transmission:`` El tipo de transmisión (Automática o Manual).
- ``fuel_type:`` El tipo de combustible que utiliza el vehículo (por ejemplo, Gasolina, Diésel, Eléctrico).
- ``drivetrain:`` El tipo de tracción del vehículo (por ejemplo, FWD - Tracción Delantera, RWD - Tracción Trasera, AWD - Tracción Total).
- ``body_type:`` El estilo de la carrocería del vehículo (por ejemplo, SUV, Sedán, Camioneta Pick-up).
- ``exterior_color:`` El color principal del exterior del vehículo.
- ``interior_color:`` El color principal del interior del vehículo.
- ``owner_count:`` El número de dueños anteriores que ha tenido el vehículo.
- ``accident_history:`` El historial de accidentes registrado del vehículo (Ninguno, Menor o Mayor).
- ``seller_type:`` El tipo de entidad que vende el vehículo (Concesionario o Particular).
- ``condition:`` La condición general del vehículo (Excelente, Buena o Regular).
- ``trim:`` El nivel de equipamiento específico del modelo del vehículo.
- ``vehicle_age:`` La antigüedad del vehículo en años, calculada como Año Actual - Year.
- ``mileage_per_year:`` El promedio de millas que el vehículo fue conducido por año.
- ``brand_popularity:`` Una puntuación que representa la popularidad de la marca según su frecuencia en el conjunto de datos.
- ``price:`` El precio de venta del vehículo usado en USD (Dólares Estadounidenses).
 
Fuente: [Automotive Price Prediction Dataset](https://www.kaggle.com/datasets/metawave/vehicle-price-prediction).

---

## 🧪 Desarrollo del Proyecto

### 1. **Carga y exploración inicial de los datos(Limpieza)**:
Como primer paso para el desarrollo del proyecto se realizo la descarga del conjunto de datos por medio del siguiente script:
```Python
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
```
Esto script nos permitio descargar el conjunto de datos Automotive Price Prediction Dataset desde kaggle.

Posteriormente se cargó el conjunto de datos en un Notebook para realizar una exploración y conocer/familiarizarse más sobre el conjunto de datos y detectar posibles problemas con el conjunto de datos.

El conjunto de datos cuenta con 1000000 registro(filas) y 20 columnas de las cuales cuentan con los siguientes tipo de datos:
```Bash
#   Column            Non-Null Count    Dtype  
---  ------            --------------    -----  
 0   make              1000000 non-null  object 
 1   model             1000000 non-null  object 
 2   year              1000000 non-null  int64  
 3   mileage           1000000 non-null  int64  
 4   engine_hp         1000000 non-null  int64  
 5   transmission      1000000 non-null  object 
 6   fuel_type         1000000 non-null  object 
 7   drivetrain        1000000 non-null  object 
 8   body_type         1000000 non-null  object 
 9   exterior_color    1000000 non-null  object 
 10  interior_color    1000000 non-null  object 
 11  owner_count       1000000 non-null  int64  
 12  accident_history  249867 non-null   object 
 13  seller_type       1000000 non-null  object 
 14  condition         1000000 non-null  object 
 15  trim              1000000 non-null  object 
 16  vehicle_age       1000000 non-null  int64  
 17  mileage_per_year  1000000 non-null  float64
 18  brand_popularity  1000000 non-null  float64
 19  price             1000000 non-null  float64
```
Como los tipos de datos de cada columna son correctos y no haya necesidad de realizar alguna transformación de datos adicional pasamos a explorar cada columna en busca de valores nulos/faltantes.

```Bash
Columnas del conjunto de datos con valores nulos
Columnas            Cant. Nulos
make                         0
model                        0
year                         0
mileage                      0
engine_hp                    0
transmission                 0
fuel_type                    0
drivetrain                   0
body_type                    0
exterior_color               0
interior_color               0
owner_count                  0
accident_history        750133
seller_type                  0
condition                    0
trim                         0
vehicle_age                  0
mileage_per_year             0
brand_popularity             0
price                        0
```
Se encontro que la columna `accident_history` es la unica columna con valores nulos. 

```Bash
Tipos de datos y cantidad de accident_history
Minor    199981
Major     49886

Cantidad de valores nulos encontrados
750133
```
Explorando más a detalle la columna `accident_history` se encontraron 2 categorias que son **Minor** con *199981* registros y **Mayor** con *49886* registros y *750133* registros con valores nulos, los valores nulos nos puede dar a entener que registros con dichos valores nulos son vehículos no tubieron accidentes por lo que se imputara los registros con valores nulos y se colocara `No Accident`.

```Bash
Cantidad de valores nulos después de imputar: 0
```
 Una vez que se han corregido los problemas con el conjunto de datos se guarda el conjuntos limpio en la siguiente dirección `../data/processed/vehicle_price.csv`.


2. **Limpieza y preprocesamiento**:
   - Manejo de valores nulos, duplicados, formatos y conversiones de fechas.

3. **Análisis exploratorio de datos (EDA)**:
   - [Ej. Distribución, correlaciones, agrupaciones, etc.]

4. **Visualización de datos**:
   - Uso de gráficos de barras, líneas, cajas, dispersión y mapas de calor.

5. **Modelado o reportes (opcional)**:
   - [Si aplica: modelos de ML, clustering, predicciones, etc.]

6. **Conclusiones y recomendaciones**:
   - Síntesis de hallazgos clave y propuestas de acción.

---

## 💡 Insights Claves

- [Insight 1]
- [Insight 2]
- [Recomendación práctica o estratégica basada en los datos]

---

## 🛠️ Tecnologías

- Python
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab
- [Otras herramientas que uses, como Scikit-learn, Plotly, etc.]

---

## ⚙️ Instalación

### 1. Clonar este repositorio:
```bash
git clone https://github.com/tu_usuario/nombre_del_proyecto.git
```
### 2. Uso de un Entorno Virtual para Aislar Dependencias

Para evitar conflictos con versiones de librerías, se recomienda usar entornos virtuales.

####  Crear y Activar un Entorno Virtual

##### Crear el entorno virtual:
```
python -m venv venv
```
##### Activar el entorno:
* #### En Windows:

    ```
    venv\Scripts\activate
    ```

* #### En Mac/Linux::

    ```
    source venv/bin/activate
    ```
#### 3. Instalar dependencias dentro del entorno:
* #### Opición 1:
    ```
    pip install -r requirements.txt
    ```

* #### Opción 2 (De forma manual):
    ```
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

---

## 👤 Autor

**Said Mariano Sánchez** – *smariano170@gmail.com*  
Este proyecto forma parte de mi portafolio como analista de datos Jr.

---

## 📝 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**. Puedes usarlo, modificarlo y distribuirlo libremente, siempre que menciones al autor original.

---