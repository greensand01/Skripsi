import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121 

tf.keras.backend.clear_session()

# =====================================================
# 0. KONFIGURASI UMUM DAN PARAMETER KHUSUS 🚀
# =====================================================
MODEL_NAME = "DenseNet121" 
OUTPUT_DIR = f"output_{MODEL_NAME.lower()}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi GPU (disarankan)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

data_dir = r"dataset" 
TARGET_SIZE = (224, 224) 
EPOCHS = 50 
LEARNING_RATE = 0.0001 
INPUT_SHAPE = TARGET_SIZE + (3,)

# =====================================================
# 1. FUNGSI UNTUK MEMBANGUN DENSENET121
# =====================================================
def build_model(input_shape, num_classes):
    base_model = DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet' 
    )
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name=MODEL_NAME)
    return model

# =====================================================
# 2. PERSIAPAN DATA DAN BOBOT KELAS
# =====================================================
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2, rotation_range=20, shear_range=0.2, 
    zoom_range=0.25, horizontal_flip=True, vertical_flip=True
)

train_generator = datagen.flow_from_directory(
    data_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='categorical', subset='training', shuffle=True,
    color_mode='rgb' 
)

val_generator = datagen.flow_from_directory(
    data_dir, target_size=TARGET_SIZE, batch_size=32, class_mode='categorical', subset='validation', shuffle=False,
    color_mode='rgb'
)

NUM_CLASSES = len(train_generator.class_indices)

# Bobot Kelas
y_labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_labels), y=y_labels
)
class_weights_dict = dict(enumerate(class_weights))


# =====================================================
# 3. FUNGSI PENYIMPANAN LENGKAP 🖼️
# =====================================================

def save_analysis_results(model, history, val_generator, output_dir):
    class_labels = list(train_generator.class_indices.keys())
    
    # RIWAYAT EPOCH KE .TXT
    history_data = history.history
    history_save_path = os.path.join(output_dir, f"{MODEL_NAME.lower()}_epoch_history.txt")

    with open(history_save_path, 'w') as f:
        header = f"{'Epoch':<5}{'Loss':<10}{'Accuracy':<10}{'Val_Loss':<10}{'Val_Accuracy':<10}\n"
        f.write(header)
        f.write("-" * 45 + "\n")
        
        for i in range(len(history_data['loss'])):
            line = f"{i+1:<5}{history_data['loss'][i]:<10.4f}{history_data['accuracy'][i]:<10.4f}{history_data['val_loss'][i]:<10.4f}{history_data['val_accuracy'][i]:<10.4f}\n"
            f.write(line)
            
    print(f"✅ Riwayat Epoch disimpan ke: {history_save_path}")

    # Grafik Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Training and Validation Accuracy ({MODEL_NAME})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Training and Validation Loss ({MODEL_NAME})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{MODEL_NAME.lower()}_akurasi_loss_plot.png"))
    plt.close()

    # Confusion Matrix & Report
    val_generator.reset()
    predictions = model.predict(val_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), xticks_rotation='vertical')
    plt.title(f"Confusion Matrix ({MODEL_NAME})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{MODEL_NAME.lower()}_confusion_matrix.png"))
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    with open(os.path.join(output_dir, f"{MODEL_NAME.lower()}_classification_report.txt"), "w") as f:
        f.write("\n--- Classification Report (Detail Performa Kelas) ---\n")
        f.write(report)
    
    print(f"✅ Semua hasil analisis {MODEL_NAME} disimpan.")
    print("\n--- Laporan Akurasi Per Kelas ---")
    print(report)


# =====================================================
# 4. EKSEKUSI TRAINING 🏁
# =====================================================

model = build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTotal Parameters ({MODEL_NAME}): {model.count_params()}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS, 
    class_weight=class_weights_dict,
    verbose=1
)

# PENYIMPANAN MODEL AKHIR
model_save_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME.lower()}_final_model.h5")
model.save(model_save_path)
print(f"\n✅ Model {MODEL_NAME} disimpan ke: {model_save_path}")

save_analysis_results(model, history, val_generator, OUTPUT_DIR)

print(f"\n--- PROGRAM {MODEL_NAME} SELESAI ---")