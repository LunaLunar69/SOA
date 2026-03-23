import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from utils_data import SOADataLoader

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' # Ruta donde esté la carpeta con los datasets
TEACHER_PATH = 'teacher_soa_resnet.keras'
WINDOW_SIZE = 64
BATCH_SIZE = 512
EPOCHS = 60 # AUMENTADO: Más tiempo de estudio para alcanzar los picos
STEPS_PER_EPOCH = 4000 

# --- 2. CARGA DE DATOS ---
loader = SOADataLoader(DATA_DIR, WINDOW_SIZE, BATCH_SIZE)
train_ds = loader.get_tf_dataset()
NUM_FEATURES = loader.num_features # Serán las 4 características

# --- 3. CARGAR TEACHER Y EXTRAER CAPAS INTERMEDIAS ---
# Definimos el lego personalizado para que Keras lo reconozca
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
        context_vector = attention_weights * inputs
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# Cargamos el modelo inyectando el lego personalizado
teacher_full = tf.keras.models.load_model(
    TEACHER_PATH, 
    compile=False,
    custom_objects={'SimpleAttention': SimpleAttention}
)
teacher_full.trainable = False

# Extraemos las salidas intermedias para el Knowledge Distillation
teacher_mlkd = models.Model(
    inputs=teacher_full.input,
    outputs=[
        teacher_full.get_layer('conv_intermediate').output,
        teacher_full.get_layer('bigru_intermediate').output,
        teacher_full.output
    ],
    name="Teacher_Features"
)

# --- 4. MODELO STUDENT ---
def build_student_model(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))
    
    # Equivalente ligero a la ADCNN del maestro
    s_cnn = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu', name='student_cnn')(inputs)
    s_pool = layers.MaxPooling1D(pool_size=2)(s_cnn)
    
    # Equivalente ligero a la BiGRU (solo GRU)
    s_gru = layers.GRU(32, return_sequences=True, name='student_gru')(s_pool)
    
    # Capas de salida
    gap = layers.GlobalAveragePooling1D()(s_gru)
    dense = layers.Dense(16, activation='relu')(gap)
    outputs = layers.Dense(1, activation='linear', name='student_output')(dense)
    
    return models.Model(inputs=inputs, outputs=[s_cnn, s_gru, outputs], name="Student_SOA")

student = build_student_model(WINDOW_SIZE, NUM_FEATURES)
student.summary()

# --- 5. CICLO DE ENTRENAMIENTO PERSONALIZADO ---
optimizer = Adam(learning_rate=0.001)
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(x, y_true):
    # Inferencia del experto
    t_cnn, t_gru, t_logits = teacher_mlkd(x, training=False)
    
    with tf.GradientTape() as tape:
        # Inferencia del aprendiz
        s_cnn, s_gru, s_logits = student(x, training=True)
        
        # Cálculos de castigo (Pérdidas)
        loss_label = mse(y_true, s_logits)
        loss_kd = mse(t_logits, s_logits)
        
        t_feat_pooled = tf.reduce_mean(t_cnn, axis=[1, 2])
        s_feat_pooled = tf.reduce_mean(s_cnn, axis=[1, 2])
        loss_feat = mse(t_feat_pooled, s_feat_pooled)
        
        # ACTUALIZADO: Forzamos al modelo a respetar más la amplitud real (0.7)
        total_loss = (0.7 * loss_label) + (0.2 * loss_kd) + (0.1 * loss_feat)
        
    gradients = tape.gradient(total_loss, student.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student.trainable_variables))
    return total_loss, loss_label, loss_kd

# --- 6. EJECUCIÓN DEL ENTRENAMIENTO ---
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progbar = tf.keras.utils.Progbar(STEPS_PER_EPOCH)
    
    for step, (x_batch, y_batch) in enumerate(train_ds):
        if step >= STEPS_PER_EPOCH: break
        
        t_loss, l_lab, l_kd = train_step(x_batch, y_batch)
        progbar.update(step + 1, values=[("loss", t_loss), ("label_err", l_lab), ("kd_err", l_kd)])

# --- 7. EXPORTACIÓN DEL MODELO LIGERO ---
student.save('student_soa_distilled.keras')
print("\n¡Entrenamiento del estudiante completado y guardado!")