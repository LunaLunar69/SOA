import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob
import os
from sklearn.model_selection import train_test_split
from utils_data import SOADataLoader 

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' # Ruta donde estan los 1500 datasets .parquet
WINDOW_SIZE = 128     
BATCH_SIZE = 512  
EPOCHS = 60
STEPS_PER_EPOCH = 1000 # Reducido a 1000 para que valide más rápido
VAL_STEPS = 200        # Pasos de validación por época

# --- 2. DIVISIÓN DE DATOS (TRAIN / VAL) ---
# Obtenemos todos los archivos y los separamos (80% train, 20% val)
all_parquet_files = glob.glob(os.path.join(DATA_DIR, '*.parquet'))
train_files, val_files = train_test_split(all_parquet_files, test_size=0.2, random_state=42)

print(f"Archivos totales: {len(all_parquet_files)}")
print(f"Archivos de entrenamiento: {len(train_files)}")
print(f"Archivos de validación: {len(val_files)}")

# Inicializamos los dos loaders con sus respectivas listas de archivos
train_loader = SOADataLoader(train_files, WINDOW_SIZE, BATCH_SIZE)
val_loader = SOADataLoader(val_files, WINDOW_SIZE, BATCH_SIZE)

train_dataset = train_loader.get_tf_dataset()
val_dataset = val_loader.get_tf_dataset()
NUM_FEATURES = train_loader.num_features

# --- 3. COMPONENTES PERSONALIZADOS ---
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
        # Aplicamos y comprimimos la secuencia sumando a lo largo del tiempo
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- 4. ARQUITECTURA DEL MODELO TEACHER ---
def build_teacher_model(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))

    # Bloque 1: CNN (Extracción local)
    c = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    c = layers.BatchNormalization()(c)
    c = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', dilation_rate=2)(c)

    # Bloque 2: BiGRU con Atención
    r = layers.Bidirectional(layers.GRU(64, return_sequences=True))(c)
    a = SimpleAttention(128)(r) # La salida ya es plana (batch_size, features)
    
    # Bloque 3: Refinamiento Residual
    dense_base = layers.Dense(128, activation='relu')(a)
    dense_base = layers.Dropout(0.2)(dense_base)
    
    res = layers.Dense(128, activation='relu')(dense_base)
    res = layers.Dense(128, activation='relu')(res)
    combined = layers.Add()([dense_base, res]) 

    # Salida
    x = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Teacher_SOA_ResNet")
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

teacher = build_teacher_model(WINDOW_SIZE, NUM_FEATURES)
teacher.summary()

# --- 5. ENTRENAMIENTO ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint('teacher_best_weights.keras', save_best_only=True, monitor='val_loss')
]

print("\nIniciando entrenamiento con mezcla dinámica y validación cruzada...")
history = teacher.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks,
    verbose=1
)

# --- 6. EXPORTACIÓN ---
teacher.save('teacher_soa_resnet.keras')
print("\n¡Entrenamiento del teacher completado y guardado!")