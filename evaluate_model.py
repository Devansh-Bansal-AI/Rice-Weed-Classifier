import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# --- STEP 1: Define Constants and Paths ---
DATASET_PATH = r'K:\Data_science\rice_weed_dataset_split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = 'rice_weed_classifier_model.h5'

# --- STEP 2: Load Model ---
print("Loading the trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# --- STEP 3: Load Test Data ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    f'{DATASET_PATH}/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
CLASS_NAMES = list(test_generator.class_indices.keys())

# --- STEP 4: Predictions ---
true_classes = test_generator.classes
true_labels = tf.keras.utils.to_categorical(true_classes, num_classes=len(CLASS_NAMES))
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# --- Confusion Matrix ---
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig("confusion_matrix_counts.png")
plt.close()

# --- Normalized Confusion Matrix ---
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig("confusion_matrix_normalized.png")
plt.close()

# --- Bar Chart (Per-class Recall) ---
report = classification_report(true_classes, predicted_classes,
                               target_names=CLASS_NAMES, output_dict=True)
class_acc = [report[c]["recall"] for c in CLASS_NAMES]
plt.figure(figsize=(8, 6))
sns.barplot(x=CLASS_NAMES, y=class_acc, palette="viridis")
plt.title("Per-class Recall (Accuracy per class)")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.savefig("bar_per_class_accuracy.png")
plt.close()

# --- ROC Curves (One-vs-Rest for each class) ---
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(true_labels[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves (One-vs-Rest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("roc_curves.png")
plt.close()

# --- Scatter Plot (Confidence for True Classes) ---
true_probs = [predictions[i, true_classes[i]] for i in range(len(true_classes))]
plt.figure(figsize=(8, 6))
plt.scatter(range(len(true_probs)), true_probs, alpha=0.6, c=true_classes, cmap='tab10')
plt.title('Prediction Confidence for True Classes')
plt.xlabel('Sample Index')
plt.ylabel('Confidence (True Class Probability)')
plt.colorbar(label='True Class Index')
plt.savefig("scatter_confidence.png")
plt.close()

print("\nAll graphs saved successfully:")
print("- confusion_matrix_counts.png")
print("- confusion_matrix_normalized.png")
print("- bar_per_class_accuracy.png")
print("- roc_curves.png")
print("- scatter_confidence.png")