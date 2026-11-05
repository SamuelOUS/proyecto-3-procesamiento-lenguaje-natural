# Proyecto 3 – Procesamiento de Lenguaje Natural

**Clasificación de emociones en texto con fine-tuning de DistilBERT**

Universidad de Medellín · Procesamiento de Lenguaje Natural  
Autor: Samuel Osorio Uribe

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<usuario>/<repositorio>/blob/main/Proyecto3_NLP_Emotion_FineTuning.ipynb)

## Descripción

Se aplica fine-tuning sobre `distilbert-base-uncased` para clasificar tweets en inglés en una de seis emociones: `sadness`, `joy`, `love`, `anger`, `fear` y `surprise`. Es un problema de **clasificación de secuencias multiclase**, útil para monitoreo de redes sociales, priorización de tickets de servicio al cliente y chatbots empáticos.

## Dataset

- **Fuente:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face), Saravia et al., 2018.
- **Entrada:** `text`, un tweet corto en inglés.
- **Salida:** `label`, una de 6 emociones.
- **Particiones oficiales:** 16.000 train, 2.000 validación y 2.000 test.
- **Observación:** las clases están desbalanceadas (`joy` y `sadness` dominan; `surprise` es la más pequeña), por eso la métrica principal es **F1 macro**.

## Contenido del notebook

1. Descripción del dataset
2. Análisis exploratorio (balance de clases, longitud de textos, ejemplos por emoción, nulos y duplicados)
3. Tokenización (WordPiece, `max_length=128`, padding dinámico)
4. Fine-tuning con `Trainer` de Hugging Face
5. Búsqueda de hiperparámetros (6 configuraciones)
6. Evaluación del mejor modelo en test, matriz de confusión, inferencia y análisis de errores
7. Conclusiones

## Experimentos

Los modelos se comparan en el conjunto de **validación**. Se varió learning rate, batch size, número de épocas y weight decay.

| Learning rate | Batch | Épocas | Weight decay | Accuracy (val) | F1 macro (val) |
|---|---|---|---|---|---|
| **5e-5** | **16** | **3** | **0.01** | **0.9460** | **0.9239** |
| 5e-5 | 32 | 3 | 0.01 | 0.9435 | 0.9220 |
| 1e-4 | 32 | 3 | 0.01 | 0.9455 | 0.9204 |
| 5e-5 | 32 | 5 | 0.10 | 0.9435 | 0.9189 |
| 2e-5 | 32 | 3 | 0.01 | 0.9350 | 0.9097 |
| 1e-5 | 32 | 3 | 0.01 | 0.9300 | 0.9048 |

Con learning rates bajos (1e-5 y 2e-5) el modelo converge lento y queda subajustado. Con 5 épocas aparece sobreajuste: la pérdida de validación sube mientras la de entrenamiento sigue bajando.

## Mejor modelo

**DistilBERT** con learning rate 5e-5, batch 16, 3 épocas y weight decay 0.01. Evaluado una sola vez en test:

| Métrica | Valor |
|---|---|
| Accuracy | 0.9325 |
| F1 macro | 0.8899 |

El rendimiento es alto en las clases mayoritarias (F1 ≥ 0.93 en `sadness`, `joy` y `anger`) y menor en las minoritarias (`love` F1 = 0.81, `surprise` F1 = 0.76), lo que refleja el efecto del desbalance de clases.

## Cómo ejecutarlo

1. Abrir el notebook en Google Colab (botón de arriba).
2. Activar GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas en orden. El entrenamiento completo toma unos 20 minutos en una T4.

No hace falta descargar el dataset ni crear cuenta en Hugging Face: se descarga automáticamente con `load_dataset`.

## Tecnologías

Python · PyTorch · Hugging Face `transformers` y `datasets` · scikit-learn · pandas · matplotlib · seaborn

## Limitaciones y trabajo futuro

- Cada configuración se entrenó con una sola semilla, por lo que las diferencias pequeñas entre configuraciones no son concluyentes.
- Las etiquetas se obtuvieron de forma semiautomática (hashtags), por lo que pueden contener ruido.
- El modelo solo funciona con texto en inglés.
- Mejoras posibles: pesos de clase en la función de pérdida, aumento de datos para las clases minoritarias, varias semillas y modelos más grandes (BERT o RoBERTa).

## Referencia

Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., Chen, Y.-S. (2018). *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP.
