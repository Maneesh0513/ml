import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# -----------------------------
# GPU Configuration
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -----------------------------
# Enhanced Parameters
# -----------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 64
NUM_CLASSES = 6
EPOCHS = 40

# -----------------------------
# Improved Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './input/plant_village/train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    './input/plant_village/test',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# Stochastic Gradient Descent with Mini-batches
# -----------------------------
def stochastic_gradient_descent(epoch):
    # Adaptive learning rate with decay as described
    initial_lr = 0.01
    decay_rate = 0.9
    # Decreasing learning rate over time to aid convergence
    return initial_lr * (decay_rate ** (epoch//5))

# -----------------------------
# Enhanced Model Architecture Based on DCNN Components
# -----------------------------
def create_improved_dcnn_model():
    model = Sequential([
        # First Convolutional Layer
        # Depth parameter: 32 filters to detect different features
        Conv2D(32, (3,3), activation='relu', padding='same', 
              input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        # Max Pooling as described for faster convergence
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Second Convolutional Layer with increased depth
        # Stride parameter implemented in convolution
        Conv2D(64, (3,3), activation='relu', padding='same', strides=(1,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Third Convolutional Layer with further increased depth
        # Zero-padding parameter implemented with 'same' padding
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # ReLU Layer is implemented in each Conv2D with activation='relu'
        
        # Fully Connected Layer as described
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        # Final output with softmax activation for classification
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Using Stochastic Gradient Descent as described in the paragraphs
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    
    # Loss layer implemented through loss function
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = create_improved_dcnn_model()
model.summary()

# -----------------------------
# Training Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
    LearningRateScheduler(stochastic_gradient_descent)  # Using our custom SGD scheduler
]

# -----------------------------
# Model Training with mini-batch SGD
# -----------------------------
history = model.fit(
    train_generator,  # Using mini-batches as defined in BATCH_SIZE
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# Visualization & Evaluation
# -----------------------------
# Plot training history
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Evolution')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Final evaluation
train_loss, train_acc = model.evaluate(train_generator, verbose=0)
val_loss, val_acc = model.evaluate(val_generator, verbose=0)

print("\n" + "="*60)
print(f"{'Final Training Accuracy:':<25} {train_acc*100:.2f}%")
print(f"{'Final Validation Accuracy:':<25} {val_acc*100:.2f}%")
print("="*60)

# Save final model
model.save(f'plant_disease_model_{val_acc*100:.1f}acc.h5')
