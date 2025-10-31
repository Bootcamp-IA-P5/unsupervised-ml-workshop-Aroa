# 🍄 Taller de Aprendizaje Automático No Supervisado con el Mushroom Dataset

## 🌟 Descripción y Objetivos

Este proyecto explora la estructura intrínseca del **Mushroom Dataset** mediante técnicas de Aprendizaje Automático No Supervisado.

Los objetivos clave son:

* Limpiar y preprocesar datos categóricos usando **One-Hot Encoding**.
* Reducir la dimensionalidad con **PCA** (Análisis de Componentes Principales).
* Aplicar **K-Means Clustering** para descubrir agrupaciones naturales ($K=2$).
* Comparar el rendimiento no supervisado (**ARI**) con una línea base supervisada (**Random Forest**).

---

## 🛠️ Tecnologías y Librerías

El proyecto se desarrolló en un entorno de **Jupyter/Google Colab** y requiere las siguientes dependencias:

### Requisitos de Python

| Categoría | Librería | Propósito Principal |
| :--- | :--- | :--- |
| **Núcleo de Datos** | `pandas` | Limpieza, OHE y manipulación de DataFrames. |
| **Cálculo Numérico** | `numpy` | Manejo de arrays y valores faltantes (`np.nan`). |
| **Modelado ML** | `scikit-learn` | PCA, K-Means, RandomForest y Métricas de Evaluación. |
| **Visualización** | `matplotlib`, `seaborn` | Generación de gráficos (PCA 2D, Método del Codo). |

### Instalación

Se recomienda instalar las dependencias en un entorno virtual:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

🚀 Guía de Ejecución
1. Carga y Preprocesamiento de Datos

El dataset se obtiene directamente desde el repositorio de la UCI:

# Carga directa del dataset (asumiendo column_names definido)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
df = pd.read_csv(url, header=None, names=column_names)

# Comandos de Preprocesamiento Clave:
# 1. Eliminación de columna 'veil-type' (constante) y filas con valores '?' en 'stalk-root'.
# 2. Aplicación de One-Hot Encoding (OHE) a 21 columnas categóricas, resultando en 96 features.
X_encoded = pd.get_dummies(X, drop_first=False) 

# 3. Escalado de Datos (Fundamental para PCA y K-Means):
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

```

¡Aquí tienes esa sección del README en formato Markdown, con las tablas bien estructuradas!

Markdown

### 2. Modelado y Evaluación

| Modelo / Técnica | Comando Principal | Métrica de Evaluación |
| :--- | :--- | :--- |
| **PCA** | `pca = PCA(n_components=2).fit_transform(X_scaled)` | Varianza Explicada. |
| **Random Forest (Baseline)** | `rf = RandomForestClassifier(...).fit(X_train, y_train)` | **Accuracy** (Precisión en test). |
| **K-Means Clustering** | `kmeans = KMeans(n_clusters=2, ...).fit_predict(X_scaled)` | **Adjusted Rand Index (ARI)**. |

---

## 📊 Resultados y Conclusión Final

### Hallazgos de Rendimiento

| Tarea | Modelo | Resultado Clave | Implicación |
| :--- | :--- | :--- | :--- |
| **Supervisado** | Random Forest | **Accuracy: $\approx 1.0000$** | El problema es trivialmente clasificable con etiquetas. |
| **No Supervisado** | K-Means | **ARI: $\approx 0.99 - 1.00$** | Coincidencia casi perfecta con las clases reales. |
| **Reducción** | PCA (2D) | Separa las clases de forma visible. | La información crucial se concentra eficientemente. |

### Conclusión

El análisis demuestra que la estructura inherente del **Mushroom Dataset** es fuertemente binaria. La alta puntuación del **Adjusted Rand Index (ARI)** valida que el **K-Means Clustering** fue capaz de **descubrir la clasificación** (comestible vs. venenoso) sin utilizar las etiquetas de entrenamiento. Esto subraya la potencia de la combinación de One-Hot Encoding, PCA y K-Means para el análisis y segmentación de datos categóricos en el ámbito no supervisado.

<p align="center">Desarrollado con ❤️ usando Python. **SparkCode** ✨</p>