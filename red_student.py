import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam

# --- 1. CONFIGURACIÓN ---
CSV_PATH = 'dataset_35gb.csv'
TEACHER_PATH = 'teacher_soa_resnet.keras'
WINDOW_SIZE = 64
BATCH_SIZE = 512
EPOCHS = 30 

# --- 2. CARGAR TEACHER Y EXTRAER CAPAS INTERMEDIAS ---
teacher_full = tf.keras.models.load_model(TEACHER_PATH, compile=False)
teacher_full.trainable = False

# Crear modelo que devuelva las salidas intermedias definidas en train_teacher.py
teacher_mlkd = models.Model(
    inputs=teacher_full.input,
    outputs=[
        teacher_full.get_layer('conv_intermediate').output,
        teacher_full.get_layer('bigru_intermediate').output,
        teacher_full.output
    ],
    name="Teacher_Features"
)

# --- 3. MODELO STUDENT ---
def build_student_model(window_size):
    inputs = layers.Input(shape=(window_size, 1))
    
    # Equivalente a la ADCNN 
    s_cnn = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu', name='student_cnn')(inputs)
    s_pool = layers.MaxPooling1D(pool_size=2)(s_cnn)
    
    # Equivalente a la BiGRU (GRU simple y menos unidades)
    s_gru = layers.GRU(32, return_sequences=True, name='student_gru')(s_pool)
    
    # Capas de salida
    gap = layers.GlobalAveragePooling1D()(s_gru)
    dense = layers.Dense(16, activation='relu')(gap)
    outputs = layers.Dense(1, activation='linear', name='student_output')(dense)
    
    # El modelo devuelve [CNN_feat, GRU_feat, Logits] para el ciclo de entrenamiento
    return models.Model(inputs=inputs, outputs=[s_cnn, s_gru, outputs], name="Student_SOA")

student = build_student_model(WINDOW_SIZE)
student.summary()

# --- 4. GENERADOR DE DATOS ---
def data_generator(file_path, window_size):
    while True:
        for chunk in pd.read_csv(file_path, chunksize=500000):
            x_data = chunk['Power_Output_mW'].values.astype(np.float32).reshape(-1, 1)
            y_data = chunk['Current_Input_A'].values.astype(np.float32).reshape(-1, 1)
            
            # Normalización
            x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data) + 1e-7)
            y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data) + 1e-7)

            for i in range(0, len(x_data) - window_size, 2): 
                yield x_data[i : i + window_size], y_data[i + window_size // 2]

train_ds = tf.data.Dataset.from_generator(
    lambda: data_generator(CSV_PATH, WINDOW_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(WINDOW_SIZE, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 5. CICLO DE ENTRENAMIENTO PERSONALIZADO ---
optimizer = Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y_true):
    # 1. Inferencia del Teacher 
    t_cnn, t_gru, t_logits = teacher_mlkd(x, training=False)
    
    with tf.GradientTape() as tape:
        # 2. Inferencia del Student
        s_cnn, s_gru, s_logits = student(x, training=True)
        
        # 3. Cálculo de Pérdidas
        # L_label: Error con el dato real
        loss_label = mse(y_true, s_logits)
        
        # L_distill: Imitar la salida final del Teacher
        loss_kd = mse(t_logits, s_logits)
        
        # L_feat: Alineación de capas intermedias
        # Usar pooling para comparar dimensiones aunque tengan distinto número de filtros
        t_feat_pooled = tf.reduce_mean(t_cnn, axis=[1, 2])
        s_feat_pooled = tf.reduce_mean(s_cnn, axis=[1, 2])
        loss_feat = mse(t_feat_pooled, s_feat_pooled)
        
        # Pérdida Total con pesos (ajustables)
        total_loss = (0.5 * loss_label) + (0.3 * loss_kd) + (0.2 * loss_feat)
        
    # 4. Optimización
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    return total_loss, loss_label, loss_kd

# --- 6. EJECUCIÓN DEL ENTRENAMIENTO ---
steps_per_epoch = 4000 

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progbar = tf.keras.utils.Progbar(steps_per_epoch)
    
    for step, (x_batch, y_batch) in enumerate(train_ds):
        if step >= steps_per_epoch: break
        
        t_loss, l_lab, l_kd = train_step(x_batch, y_batch)
        progbar.update(step + 1, values=[("loss", t_loss), ("label_err", l_lab), ("kd_err", l_kd)])

# --- 7. EXPORTACIÓN DEL MODELO LIGERO ---
student.save('student_soa_distilled.keras')
