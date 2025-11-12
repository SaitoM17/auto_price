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

1. **Carga y exploración inicial de los datos**:
   - Exploración básica con `.head()`, `.info()`, `.describe()`, etc.

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