import numpy as np
import tensorflow as tf
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import json
from utils_data import SOADataLoader, TARGET

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' # Ruta donde esté la carpeta con los datasets
STUDENT_PATH = 'student_soa_distilled.keras'
SCALERS_PATH = 'scalers.json'
WINDOW_SIZE = 64
NUM_SAMPLES_TO_TEST = 1000 # Cuántos puntos de la señal queremos graficar

# --- 2. CARGAR EL MODELO Y LOS SCALERS ---
print("Cargando el modelo destilado...")
model = tf.keras.models.load_model(STUDENT_PATH, compile=False)

print("Cargando scalers...")
with open(SCALERS_PATH, 'r') as f:
    scalers = json.load(f)

def denormalize_target(scaled_value):
    """Convierte el valor (0 a 1) de regreso a Amperios reales (Current_Input_A)"""
    v_min = scalers['min'][TARGET[0]]
    v_max = scalers['max'][TARGET[0]]
    return scaled_value * (v_max - v_min + 1e-7) + v_min

# --- 3. OBTENER DATOS DE PRUEBA ---
print("Preparando datos de prueba...")
# Reutilizamos tu DataLoader para sacar un lote de datos
loader = SOADataLoader(DATA_DIR, WINDOW_SIZE, batch_size=NUM_SAMPLES_TO_TEST)
test_dataset = loader.get_tf_dataset()

# Extraemos solo un lote (batch) para evaluar y graficar
x_test, y_test_scaled = next(iter(test_dataset))

# --- 4. HACER PREDICCIONES ---
print("Realizando predicciones...")
# Recordar que el modelo Student devuelve [CNN_feat, GRU_feat, Logits]. 
# Solo nos interesan las predicciones finales (Logits), que es el índice 2.
_, _, predictions_scaled = model.predict(x_test)

# --- 5. DESNORMALIZAR DATOS ---
y_test_real = denormalize_target(y_test_scaled.numpy().flatten())
predictions_real = denormalize_target(predictions_scaled.flatten())

# --- 6. CALCULAR MÉTRICAS ---
mse = np.mean(np.square(y_test_real - predictions_real))
mae = np.mean(np.abs(y_test_real - predictions_real))
print("-" * 30)
print(f"Resultados de la Ecualización SOA:")
print(f"Error Cuadrático Medio (MSE): {mse:.6f} A^2")
print(f"Error Absoluto Medio (MAE): {mae:.6f} A")
print("-" * 30)

# --- 7. GRAFICAR RESULTADOS ---
plt.figure(figsize=(14, 6))
plt.plot(y_test_real, label='Señal Original (Amperios)', color='blue', alpha=0.7)
plt.plot(predictions_real, label='Señal Ecualizada (Student)', color='red', linestyle='dashed', alpha=0.9)

plt.title('Reconstrucción de la Señal: Original vs Ecualizada')
plt.xlabel('Muestras de Tiempo')
plt.ylabel('Current Input (A)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar la gráfica en una imagen
plt.savefig('resultado_ecualizacion.png', dpi=300)
print("¡Gráfica guardada como 'resultado_ecualizacion.png'!")
plt.show()