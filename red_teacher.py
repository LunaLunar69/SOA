import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from utils_data import SOADataLoader

# --- 1. CONFIGURACIÓN ---
DATA_DIR = 'datasets' # Ruta donde esté la carpeta con los datasets
WINDOW_SIZE = 64  # Ajustar según necesidad
BATCH_SIZE = 512  
EPOCHS = 60
STEPS_PER_EPOCH = 5000 

# --- 2. CARGA DE DATOS ---
loader = SOADataLoader(DATA_DIR, WINDOW_SIZE, BATCH_SIZE)
train_dataset = loader.get_tf_dataset()
NUM_FEATURES = loader.num_features # Ahora son 4

# --- 3. COMPONENTES PERSONALIZADOS ---
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

# --- 4. ARQUITECTURA DEL MODELO TEACHER ---
def build_teacher_model(window_size, num_features):
    # ¡Actualizado para recibir las 4 características!
    inputs = layers.Input(shape=(window_size, num_features))

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

teacher = build_teacher_model(WINDOW_SIZE, NUM_FEATURES)
teacher.summary()

# --- 5. ENTRENAMIENTO ---
callbacks = [
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1),
    ModelCheckpoint('teacher_best_weights.keras', save_best_only=True, monitor='loss')
]

history = teacher.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    verbose=1
)

# --- 6. EXPORTACIÓN ---
teacher.save('teacher_soa_resnet.keras')
print("\n¡Entrenamiento del teacher completado y guardado!")