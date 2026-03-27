import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import glob
import os
from tensorflow.keras import layers
from utils_data import SOADataLoader # O utils_data si así se llama tu archivo

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' 
STUDENT_PATH = 'student_soa_best.keras' # Usamos el que se guardó con el Early Stopping
SCALERS_PATH = 'scalers.json'
WINDOW_SIZE = 128 # Sincronizado con el entrenamiento
NUM_SAMPLES_TO_TEST = 1000 # Muestras a graficar
TARGET = ['Current_Input_A']

# --- 2. CAPA PERSONALIZADA (Obligatoria para que Keras entienda el modelo) ---
@tf.keras.utils.register_keras_serializable()
class SimpleAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = layers.Dense(self.units, activation='tanh')
        self.V = layers.Dense(1)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        score = self.V(self.W(inputs))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- 3. CARGAR EL MODELO Y LOS SCALERS ---
print("Cargando el modelo estudiante destilado (limpio)...")
model = tf.keras.models.load_model(
    STUDENT_PATH, 
    compile=False,
    custom_objects={'SimpleAttention': SimpleAttention}
)

print("Cargando scalers...")
with open(SCALERS_PATH, 'r') as f:
    scalers = json.load(f)

def denormalize_target(scaled_value):
    """Convierte el valor (0 a 1) de regreso a Amperios reales"""
    v_min = scalers['min'][TARGET[0]]
    v_max = scalers['max'][TARGET[0]]
    return scaled_value * (v_max - v_min + 1e-7) + v_min

# --- 4. OBTENER DATOS DE PRUEBA ---
print("Preparando datos de prueba...")
# Obtenemos todos los archivos y tomamos los últimos 3 para sacar una muestra variada
all_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
test_files = all_files[-3:] 

# Usamos nuestro loader actualizado
loader = SOADataLoader(test_files, WINDOW_SIZE, batch_size=NUM_SAMPLES_TO_TEST)
test_dataset = loader.get_tf_dataset()

# Extraemos solo un lote (batch) de 1000 ventanas para evaluar
x_test, y_test_scaled = next(iter(test_dataset))

# --- 5. HACER PREDICCIONES ---
print("Realizando predicciones...")
# Como guardamos la versión de inferencia, el predict escupe los datos directamente
predictions_scaled = model.predict(x_test)

# --- 6. DESNORMALIZAR DATOS ---
y_test_real = denormalize_target(y_test_scaled.numpy().flatten())
predictions_real = denormalize_target(predictions_scaled.flatten())

# --- 7. CALCULAR MÉTRICAS ---
mse = np.mean(np.square(y_test_real - predictions_real))
mae = np.mean(np.abs(y_test_real - predictions_real))
print("-" * 30)
print(f"Resultados de la Ecualización SOA (KD Student):")
print(f"Error Cuadrático Medio (MSE): {mse:.6e} A^2")
print(f"Error Absoluto Medio (MAE): {mae:.6e} A")
print("-" * 30)

# --- 8. GRAFICAR RESULTADOS ---
plt.figure(figsize=(14, 6))
# Graficamos solo 200 puntos para que la señal sea visible y no un bloque de color
PUNTOS_GRAFICA = 200 
plt.plot(y_test_real[:PUNTOS_GRAFICA], label='Señal Original Ideal (Amperios)', color='#1f77b4', linewidth=2)
plt.plot(predictions_real[:PUNTOS_GRAFICA], label='Señal Ecualizada por el Student', color='#ff7f0e', linestyle='--', linewidth=2)

plt.title('Desempeño del Ecualizador SOA (Knowledge Distillation)')
plt.xlabel('Muestras de Tiempo')
plt.ylabel('Current Input (A)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Guardar la gráfica en una imagen
plt.savefig('resultado_ecualizacion.png', dpi=300)
print("¡Gráfica guardada como 'resultado_ecualizacion.png'!")
plt.show()
