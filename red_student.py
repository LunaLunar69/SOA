import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import glob
import os
from sklearn.model_selection import train_test_split
from utils_data  import SOADataLoader 

# --- 1. CONFIGURACIÓN ---
# CAMBIO: Apuntando a la nueva data y al nuevo maestro
DATA_DIR = 'datasets' 
TEACHER_PATH = 'teacher_soa5_best.keras'
WINDOW_SIZE = 128 
BATCH_SIZE = 512
EPOCHS = 60       
STEPS_PER_EPOCH = 1000 
VAL_STEPS = 200

# Validaciones de seguridad
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Falta la carpeta {DATA_DIR}. Corre primero el generador.")
if not os.path.exists(TEACHER_PATH):
    raise FileNotFoundError(f"Falta el modelo {TEACHER_PATH}. Entrena primero al Teacher.")

# --- 2. DIVISIÓN DE DATOS (TRAIN / VAL) ---
all_parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
train_files, val_files = train_test_split(all_parquet_files, test_size=0.2, random_state=42)

print(f"Archivos de entrenamiento: {len(train_files)} | Archivos de validación: {len(val_files)}")

train_loader = SOADataLoader(train_files, WINDOW_SIZE, BATCH_SIZE)
val_loader = SOADataLoader(val_files, WINDOW_SIZE, BATCH_SIZE)

train_ds = train_loader.get_tf_dataset()
val_ds = val_loader.get_tf_dataset()
NUM_FEATURES = train_loader.num_features

# --- 3. COMPONENTE PERSONALIZADO ---
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
    s_pool = layers.MaxPooling1D(pool_size=2)(s_cnn)
    
    # GRU ligera
    s_gru = layers.GRU(32, return_sequences=True)(s_pool)
    
    # Atención miniatura
    s_att = SimpleAttention(32)(s_gru)
    
    # Capas densas
    dense = layers.Dense(16, activation='relu')(s_att)
    outputs = layers.Dense(1, activation='linear')(dense)
    
    return models.Model(inputs=inputs, outputs=[s_cnn, outputs], name="Student_SOA")

student = build_student_model(WINDOW_SIZE, NUM_FEATURES)
student.summary()

# --- 6. FUNCIONES DE ENTRENAMIENTO PERSONALIZADAS ---
optimizer = Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y_true):
    t_cnn, t_logits = teacher_mlkd(x, training=False)
    
    with tf.GradientTape() as tape:
        s_cnn, s_logits = student(x, training=True)
        
        loss_label = mse(y_true, s_logits)
        loss_kd = mse(t_logits, s_logits)
        
        t_feat_pooled = tf.reduce_mean(t_cnn, axis=[1, 2])
        s_feat_pooled = tf.reduce_mean(s_cnn, axis=[1, 2])
        loss_feat = mse(t_feat_pooled, s_feat_pooled)
        
        total_loss = (0.4 * loss_label) + (0.4 * loss_kd) + (0.2 * loss_feat)
        
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
patience = 10 
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
    
    # Lógica de Guardado
    if val_loss_epoch < best_val_loss:
        print(f"¡Mejora detectada! ({best_val_loss:.4f} -> {val_loss_epoch:.4f}). Guardando modelo...")
        best_val_loss = val_loss_epoch
        
        inference_student = models.Model(inputs=student.input, outputs=student.output[1])
        # CAMBIO: Nuevo nombre para proteger el anterior
        inference_student.save('student_soa5_best.keras')
        wait = 0 
    else:
        wait += 1
        print(f"Sin mejora desde hace {wait} épocas.")
        if wait >= patience:
            print(f"\n¡Early Stopping activado! El modelo alcanzó su límite en la época {epoch+1}.")
            break

print("\n¡Entrenamiento finalizado exitosamente!")