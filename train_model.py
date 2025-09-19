import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# --- STEP 3: Load and Preprocess the Data ---

# Define the path to your newly segregated dataset
DATASET_PATH = r'K:\Data_science\rice_weed_dataset_split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation for training to improve model generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation and test sets
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from the directories
train_generator = train_datagen.flow_from_directory(
    f'{DATASET_PATH}/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    f'{DATASET_PATH}/validation',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    f'{DATASET_PATH}/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes from the generator
NUM_CLASSES = train_generator.num_classes
print(f"\nNumber of classes: {NUM_CLASSES}")

# --- STEP 4: Choose and Build the Model (Transfer Learning) ---

# Load the pre-trained MobileNetV2 model without its top (classification) layer
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add new custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# --- STEP 5: Train the Model ---

print("\nStarting model training...")
# You can adjust the number of epochs (passes through the dataset)
EPOCHS = 10
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the trained model for future use
model.save('rice_weed_classifier_model.h5')
print("\nModel saved as 'rice_weed_classifier_model.h5'")


# --- STEP 6: Evaluate the Final Output ---

print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Optional: Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()