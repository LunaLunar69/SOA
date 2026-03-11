import pandas as pd
import tensorflow as tf
import numpy as np
import joblib  # Para guardar los scalers
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils_data import SOADataLoader
from params import SOAparams 

p = SOAparams()

loader = SOADataLoader('dataset_35gb.csv', WINDOW_SIZE, BATCH_SIZE)

# --- 1. CONFIGURACIÓN ---
CSV_PATH = 'dataset_35gb.csv' # Dataset
WINDOW_SIZE = p.samples_per_bit * 4  # Ajustar según p.samples_per_bit * 4
BATCH_SIZE = 512  
EPOCHS = 60

# --- 2. COMPONENTES PERSONALIZADOS ---
@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))
    mae = K.mean(K.abs(y_true - y_pred))
    return 0.7 * mse + 0.3 * mae

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

# --- 3. GENERADOR DE DATOS ---
def data_generator(file_path, window_size, batch_size):
    """Generador para leer el CSV por trozos para no agotar la RAM."""
    # Nota: Los scalers deben calcularse antes o usar valores fijos
    while True: # Reinicia el archivo al terminar una época
        for chunk in pd.read_csv(file_path, chunksize=500000):
            # Preprocesamiento básico
            x_data = chunk['Power_Output_mW'].values.astype(np.float32).reshape(-1, 1)
            y_data = chunk['Current_Input_A'].values.astype(np.float32).reshape(-1, 1)
            
            # Normalización 
            x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data) + 1e-7)
            y_data = (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data) + 1e-7)

            for i in range(len(x_data) - window_size):
                yield x_data[i : i + window_size], y_data[i + window_size // 2]

# Crear el pipeline de datos optimizado
output_signature = (
    tf.TensorSpec(shape=(WINDOW_SIZE, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(CSV_PATH, WINDOW_SIZE, BATCH_SIZE),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_dataset = loader.get_tf_dataset()


# --- 4. ARQUITECTURA DEL MODELO TEACHER ---
def build_teacher_model(window_size):
    inputs = layers.Input(shape=(window_size, 1))

    # Bloque 1: ADCNN (Dilatación)
    c = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    c = layers.BatchNormalization()(c)
    c = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', dilation_rate=2, name='conv_intermediate')(c)
    c = layers.MaxPooling1D(pool_size=2)(c) 

    # Bloque 2: BiGRU con Atención
    r = layers.Bidirectional(layers.GRU(64, return_sequences=True), name='bigru_intermediate')(c)
    a = SimpleAttention(128)(r)
    a = layers.GlobalAveragePooling1D()(a)

    # Bloque 3: Refinamiento Residual
    dense_base = layers.Dense(64, activation='relu')(a)
    res = layers.Dense(64, activation='relu')(dense_base)
    res = layers.Dense(64, activation='relu')(res)
    combined = layers.Add()([dense_base, res])

    # Salida
    x = layers.Dense(32, activation='relu')(combined)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Teacher_SOA_ResNet")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss)
    return model

# --- 5. ENTRENAMIENTO ---
teacher.fit(train_dataset, steps_per_epoch=5000, ...)

teacher = build_teacher_model(WINDOW_SIZE)
teacher.summary()



# Callbacks
callbacks = [
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint('teacher_best_weights.keras', save_best_only=True, monitor='loss')
]

# Entrenamiento
# Nota: 'steps_per_epoch' es necesario cuando usas generadores infinitos
# Estimación: (Total filas / BATCH_SIZE)
STEPS_PER_EPOCH = 5000 

history = teacher.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    verbose=1
)

# --- 6. EXPORTACIÓN ---
teacher.save('teacher_soa_resnet.keras')

