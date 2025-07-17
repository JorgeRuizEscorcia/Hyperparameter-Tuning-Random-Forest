# 🎯 Hyperparameter Tuning - Random Forest (Regresión)

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
