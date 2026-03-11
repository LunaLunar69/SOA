import pandas as pd
import numpy as np
import tensorflow as tf
import os

# Script obtener Min/Max 
import pandas as pd
chunk_iter = pd.read_csv('dataset_35gb.csv', chunksize=1000000)
c_min, c_max = float('inf'), float('-inf')

for chunk in chunk_iter:
    c_min = min(c_min, chunk['Power_Output_mW'].min())
    c_max = max(c_max, chunk['Power_Output_mW'].max())
print(f"Min: {c_min}, Max: {c_max}")

class SOADataLoader:
    def __init__(self, csv_path, window_size, batch_size=512):
        self.csv_path = csv_path
        self.window_size = window_size
        self.batch_size = batch_size
        
        # Valores de normalización pre-calculados para evitar cargar el datset completo
        # O ejecuta una pasada rápida para obtenerlos.
        self.x_min, self.x_max = 0.0, 100.0  # Power_Output_mW
        self.y_min, self.y_max = 0.0, 0.5    # Current_Input_A
        
    def _normalize(self, data, v_min, v_max):
        return (data - v_min) / (v_max - v_min + 1e-7)

    def get_generator(self):
        """Generador robusto para datasets masivos."""
        def generator():
            # Usamos un chunksize grande para lectura eficiente en SSD de RunPod
            for chunk in pd.read_csv(self.csv_path, chunksize=1000000):
                # Conversión a float32 para ahorrar memoria y usar TensorCores
                x_raw = chunk['Power_Output_mW'].values.astype(np.float32).reshape(-1, 1)
                y_raw = chunk['Current_Input_A'].values.astype(np.float32).reshape(-1, 1)
                
                # Normalización consistente
                x_scaled = self._normalize(x_raw, self.x_min, self.x_max)
                y_scaled = self._normalize(y_raw, self.y_min, self.y_max)
                
                # Creación de ventanas (Sliding Window)
                # i + window_size es el límite superior
                for i in range(len(x_scaled) - self.window_size):
                    # X: Ventana de datos
                    # y: El valor en el centro o al final de la ventana
                    yield x_scaled[i : i + self.window_size], y_scaled[i + self.window_size // 2]
                    
        return generator

    def get_tf_dataset(self):
        """Crea un pipeline de datos de alto rendimiento."""
        output_signature = (
            tf.TensorSpec(shape=(self.window_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
        
        ds = tf.data.Dataset.from_generator(
            self.get_generator(),
            output_signature=output_signature
        )
        
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE) # Preparar el siguiente batch mientras la GPU entrena
        return ds

def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: No se encuentra el archivo {path} en /workspace")