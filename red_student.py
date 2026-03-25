import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import glob
import os
from sklearn.model_selection import train_test_split
from utils_data  import SOADataLoader 

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' 
TEACHER_PATH = 'teacher_soa_resnet.keras'
WINDOW_SIZE = 128 # Sincronizado con el Teacher
BATCH_SIZE = 512
EPOCHS = 60       
STEPS_PER_EPOCH = 1000 
VAL_STEPS = 200

# --- 2. DIVISIÓN DE DATOS (TRAIN / VAL) ---
all_parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
train_files, val_files = train_test_split(all_parquet_files, test_size=0.2, random_state=42)

print(f"Archivos de entrenamiento: {len(train_files)} | Archivos de validación: {len(val_files)}")

train_loader = SOADataLoader(train_files, WINDOW_SIZE, BATCH_SIZE)
val_loader = SOADataLoader(val_files, WINDOW_SIZE, BATCH_SIZE)

train_ds = train_loader.get_tf_dataset()
val_ds = val_loader.get_tf_dataset()
NUM_FEATURES = train_loader.num_features

# --- 3. COMPONENTE PERSONALIZADO (El mismo del Teacher) ---
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
        # Suma temporal para comprimir la secuencia
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- 4. CARGA DEL TEACHER Y EXTRACCIÓN DE CAPAS ---
print("\nCargando modelo experto (Teacher)...")
teacher_full = tf.keras.models.load_model(
    TEACHER_PATH, 
    compile=False,
    custom_objects={'SimpleAttention': SimpleAttention}
)
teacher_full.trainable = False

# Extraemos la primera capa Conv1D y la salida final para la destilación
teacher_mlkd = models.Model(
    inputs=teacher_full.input,
    outputs=[
        teacher_full.layers[1].output, # Mapas de características espaciales
        teacher_full.output            # Predicción final (Soft Labels)
    ],
    name="Teacher_Features"
)

# --- 5. MODELO STUDENT (Super Ligero) ---
def build_student_model(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))
    
    # CNN ligera
    s_cnn = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    # Reducimos la secuencia a la mitad para ahorrar muchísima memoria en la GRU
    s_pool = layers.MaxPooling1D(pool_size=2)(s_cnn)
    
    # GRU ligera (solo 32 unidades)
    s_gru = layers.GRU(32, return_sequences=True)(s_pool)
    
    # Atención miniatura
    s_att = SimpleAttention(32)(s_gru)
    
    # Capas densas de salida
    dense = layers.Dense(16, activation='relu')(s_att)
    outputs = layers.Dense(1, activation='linear')(dense)
    
    # El modelo devuelve la Conv1D (para KD de características) y la predicción final
    return models.Model(inputs=inputs, outputs=[s_cnn, outputs], name="Student_SOA")

student = build_student_model(WINDOW_SIZE, NUM_FEATURES)
student.summary()

# --- 6. FUNCIONES DE ENTRENAMIENTO PERSONALIZADAS ---
optimizer = Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y_true):
    # El maestro da sus respuestas (no aprende, solo evalúa)
    t_cnn, t_logits = teacher_mlkd(x, training=False)
    
    with tf.GradientTape() as tape:
        # El aprendiz da sus respuestas
        s_cnn, s_logits = student(x, training=True)
        
        # 1. Pérdida Label: ¿Qué tan lejos está de los datos reales?
        loss_label = mse(y_true, s_logits)
        
        # 2. Pérdida KD: ¿Qué tan lejos está de la respuesta del maestro?
        loss_kd = mse(t_logits, s_logits)
        
        # 3. Pérdida de Características: ¿Extrae la información espacial igual que el maestro?
        t_feat_pooled = tf.reduce_mean(t_cnn, axis=[1, 2])
        s_feat_pooled = tf.reduce_mean(s_cnn, axis=[1, 2])
        loss_feat = mse(t_feat_pooled, s_feat_pooled)
        
        # Suma ponderada de las pérdidas
        total_loss = (0.4 * loss_label) + (0.4 * loss_kd) + (0.2 * loss_feat)
        
    # Aplicar gradientes (aprender)
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    return total_loss, loss_label, loss_kd

@tf.function
def val_step(x, y_true):
    _, t_logits = teacher_mlkd(x, training=False)
    _, s_logits = student(x, training=False)
    
    v_loss_label = mse(y_true, s_logits)
    v_loss_kd = mse(t_logits, s_logits)
    v_total = (0.5 * v_loss_label) + (0.5 * v_loss_kd)
    return v_total

# --- 7. CICLO DE ENTRENAMIENTO CON EARLY STOPPING ---
print("\nIniciando Destilación de Conocimiento (KD)...")

best_val_loss = float('inf')
patience = 10  # Épocas de tolerancia si no hay mejora
wait = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progbar = tf.keras.utils.Progbar(STEPS_PER_EPOCH)
    
    # Fase de Entrenamiento
    for step, (x_batch, y_batch) in enumerate(train_ds):
        if step >= STEPS_PER_EPOCH: break
        t_loss, l_lab, l_kd = train_step(x_batch, y_batch)
        progbar.update(step + 1, values=[("loss", t_loss), ("label_err", l_lab), ("kd_err", l_kd)])
        
    # Fase de Validación
    val_loss_epoch = 0.0
    for step, (x_val, y_val) in enumerate(val_ds):
        if step >= VAL_STEPS: break
        v_loss = val_step(x_val, y_val)
        val_loss_epoch += v_loss
        
    val_loss_epoch = val_loss_epoch / VAL_STEPS
    print(f" -> val_loss general: {val_loss_epoch:.4f}")
    
    # Lógica de Guardado y Early Stopping
    if val_loss_epoch < best_val_loss:
        print(f"¡Mejora detectada! ({best_val_loss:.4f} -> {val_loss_epoch:.4f}). Guardando modelo...")
        best_val_loss = val_loss_epoch
        
        # Guardamos un modelo limpio que solo necesita la entrada y escupe la predicción
        inference_student = models.Model(inputs=student.input, outputs=student.output[1])
        inference_student.save('student_soa_best.keras')
        wait = 0 # Reiniciamos la paciencia
    else:
        wait += 1
        print(f"Sin mejora desde hace {wait} épocas.")
        if wait >= patience:
            print(f"\n¡Early Stopping activado! El modelo alcanzó su límite en la época {epoch+1}.")
            break

print("\n¡Entrenamiento finalizado exitosamente!")