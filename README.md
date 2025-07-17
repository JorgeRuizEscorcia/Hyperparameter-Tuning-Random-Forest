# 🎯 Hyperparameter Tuning - Random Forest (Regresión)

# 🌳 Hyperparameter Tuning Random Forest

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)](https://jupyter.org/)

## 📋 Descripción del Proyecto

Este proyecto implementa técnicas avanzadas de **optimización de hiperparámetros** para modelos Random Forest usando Python y scikit-learn. El objetivo es maximizar el rendimiento del modelo mediante la exploración sistemática del espacio de hiperparámetros.

### 🎯 Objetivos
- Optimizar hiperparámetros clave de Random Forest
- Comparar diferentes estrategias de búsqueda (Grid Search, Random Search)
- Evaluar el impacto de cada hiperparámetro en el rendimiento
- Implementar validación cruzada para resultados robustos

## 🔧 Tecnologías Utilizadas

- **Python 3.7+**
- **scikit-learn** - Algoritmos de machine learning
- **pandas** - Manipulación de datos
- **numpy** - Operaciones numéricas
- **matplotlib/seaborn** - Visualización
- **jupyter notebook** - Desarrollo interactivo

## 📊 Hiperparámetros Optimizados

### 🌲 Parámetros del Bosque
- `n_estimators`: Número de árboles en el bosque
- `max_depth`: Profundidad máxima de cada árbol
- `min_samples_split`: Mínimo de muestras requeridas para dividir un nodo
- `min_samples_leaf`: Mínimo de muestras requeridas en una hoja
- `max_features`: Número de características consideradas en cada división

### ⚡ Parámetros de Rendimiento
- `bootstrap`: Uso de bootstrap para construir árboles
- `oob_score`: Cálculo del score out-of-bag
- `random_state`: Semilla para reproducibilidad

## 🚀 Metodología

### 1. **Preparación de Datos**
```python
# Carga y preprocesamiento de datos
# División en conjuntos de entrenamiento y prueba
# Normalización si es necesario
```

### 2. **Búsqueda de Hiperparámetros**
- **Grid Search**: Búsqueda exhaustiva en rejilla
- **Random Search**: Búsqueda aleatoria (más eficiente)
- **Validación Cruzada**: 5-fold CV para evaluación robusta

### 3. **Evaluación de Resultados**
- Métricas de clasificación: Accuracy, Precision, Recall, F1-score
- Matriz de confusión
- Curvas ROC y AUC
- Análisis de importancia de características

## 📈 Resultados Esperados

### Antes de la Optimización
```
Accuracy: ~XX%
F1-Score: ~XX%
Tiempo de entrenamiento: XX segundos
```

### Después de la Optimización
```
Accuracy: ~XX% (+XX% mejora)
F1-Score: ~XX% (+XX% mejora)
Tiempo de entrenamiento: XX segundos
```

## 🎨 Visualizaciones Incluidas

- **Gráfico de barras**: Comparación de métricas antes/después
- **Heatmap**: Matriz de confusión optimizada
- **Gráfico de líneas**: Evolución del rendimiento durante búsqueda
- **Gráfico de importancia**: Características más relevantes

## 🔄 Estructura del Proyecto

```
📁 Hyperparameter-Tuning-Random-Forest/
├── 📄 README.md
├── 📓 hyperparameter_tuning_rf.ipynb
├── 📊 data/
│   └── dataset.csv
├── 📈 results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── performance_comparison.png
└── 📋 requirements.txt
```



## 🎓 Aplicaciones Prácticas

Este proyecto es útil para:
- **Científicos de datos** que buscan optimizar modelos Random Forest
- **Estudiantes** aprendiendo sobre hyperparameter tuning
- **Profesionales** implementando ML en producción
- **Investigadores** comparando estrategias de optimización

## 🔍 Insights Clave

1. **n_estimators**: Más árboles mejoran rendimiento pero aumentan tiempo de cómputo
2. **max_depth**: Controla overfitting; valores muy altos pueden causar sobreajuste
3. **min_samples_split**: Valores más altos previenen overfitting
4. **max_features**: 'sqrt' suele ser óptimo para clasificación

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas para mejorar el proyecto:

1. Fork el repositorio
2. Crea una nueva rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -am 'Añade nueva mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

## 📞 Contacto

**Jorge Ruiz Escorcia**
- 📧 Email: jorgeruizescorcia@gmail.com
- 💼 LinkedIn: [jorge-ruiz-escorcia](https://linkedin.com/in/jorge-ruiz-escorcia)
- 🐙 GitHub: [JorgeRuizEscorcia](https://github.com/JorgeRuizEscorcia)

---

⭐ Si este proyecto te fue útil, ¡dale una estrella en GitHub!

💡 **¿Tienes un proyecto que necesita optimización de modelos ML?** Contacta conmigo para consultoría freelance.

Este proyecto aplica técnicas de aprendizaje automático supervisado (regresión) para predecir el puntaje global del examen Saber Pro a partir de competencias genéricas. Se utiliza un modelo de Random Forest con ajuste de hiperparámetros mediante `GridSearchCV`.

---

## 📌 Objetivo

Construir y optimizar un modelo de regresión con Random Forest para predecir el puntaje global (`SP Global`) de estudiantes universitarios, usando variables como:
- Lógica Cuantitativa (`ECG LC`)
- Lectura Crítica (`ECG RC`)
- Competencias Ciudadanas (`ECG CC`)
- Comunicación Escrita (`ECG CE`)
- Global específicas

---

## 📁 Dataset

- Fuente: Archivo Excel `Saber_Pro_comp.xlsx` almacenado en Google Drive.
- Número de variables: 6 (incluyendo la variable objetivo).
- Se realiza limpieza previa eliminando valores nulos y columnas no relevantes (`ID`).

---

## 🔧 Tecnologías y Librerías

- Python 3
- Pandas y NumPy (análisis de datos)
- Scikit-learn (`RandomForestRegressor`, `GridSearchCV`, `train_test_split`)
- Google Colab (para ejecución con acceso a Google Drive)

---

## 📈 Flujo del modelo

1. Carga de datos desde Google Drive con `pandas`.
2. Limpieza y selección de variables predictoras.
3. Separación en conjunto de entrenamiento y prueba.
4. Entrenamiento inicial con Random Forest básico.
5. Ajuste de hiperparámetros con `GridSearchCV` (validación cruzada de 5 pliegues).
6. Evaluación del modelo final:
   - Predicciones (`y_pred`)
   - Comparación con datos reales (`y_test`)
   - Cálculo de errores individuales (`diferencia`)

---

## 🔍 Hiperparámetros evaluados

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
