import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import pandas as pd

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

# Set a professional style for all plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.style.use("ggplot")

# --- Confusion Matrix ---
cm = confusion_matrix(true_classes, predicted_classes)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linecolor='black', linewidths=0.5)
plt.title('Normalized Confusion Matrix: Prediction Ratios', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix_normalized_enhanced.png")
plt.close()

# --- Classification Report Heatmap ---
report_dict = classification_report(true_classes, predicted_classes,
                                    target_names=CLASS_NAMES, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.drop(columns=['support'])
plt.figure(figsize=(12, 6))
sns.heatmap(report_df.iloc[:-3, :].T, annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, linecolor='black')
plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Classes', fontsize=14)
plt.tight_layout()
plt.savefig("classification_report_heatmap.png")
plt.close()

# --- Bar Chart (Per-class Recall) ---
report = classification_report(true_classes, predicted_classes,
                               target_names=CLASS_NAMES, output_dict=True)
class_recall = [report[c]["recall"] for c in CLASS_NAMES]
plt.figure(figsize=(12, 7))
ax = sns.barplot(x=CLASS_NAMES, y=class_recall, palette="viridis")
plt.title("Per-class Recall", fontsize=16, fontweight='bold')
plt.ylabel("Recall Score", fontsize=14)
plt.ylim(0, 1.1)
plt.xlabel("Class", fontsize=14)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=12)
plt.tight_layout()
plt.savefig("bar_per_class_accuracy_enhanced.png")
plt.close()

# --- ROC Curves (One-vs-Rest) ---
plt.figure(figsize=(12, 10))
colors = plt.cm.get_cmap('tab10', len(CLASS_NAMES))
for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(true_labels[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})", color=colors(i), linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Guess')
plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("roc_curves_enhanced.png")
plt.close()

# --- Scatter Plot (Confidence for True Classes) ---
true_probs = [predictions[i, true_classes[i]] for i in range(len(true_classes))]
plt.figure(figsize=(12, 7))
sns.scatterplot(x=range(len(true_probs)), y=true_probs, hue=true_classes,
                palette=sns.color_palette("tab10", n_colors=len(CLASS_NAMES)),
                alpha=0.6, s=50)
plt.title('Prediction Confidence for True Classes', fontsize=16, fontweight='bold')
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Confidence (True Class Probability)', fontsize=14)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, CLASS_NAMES, title='True Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(-0.05, 1.05)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("scatter_confidence_enhanced.png")
plt.close()

# --- NEW: Violin Plot (Prediction Confidence Distribution) ---
plt.figure(figsize=(12, 7))
confidence_df = pd.DataFrame(predictions, columns=CLASS_NAMES)
confidence_df["True Class"] = [CLASS_NAMES[i] for i in true_classes]
sns.violinplot(data=confidence_df.melt(id_vars="True Class", var_name="Predicted Class", value_name="Confidence"),
               x="Predicted Class", y="Confidence", palette="Set2", cut=0, inner="quartile")
plt.title("Prediction Confidence Distribution (Violin Plot)", fontsize=16, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("Confidence Score", fontsize=14)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("violin_confidence_distribution.png")
plt.close()

print("\nAll enhanced graphs saved successfully:")
print("- confusion_matrix_normalized_enhanced.png")
print("- classification_report_heatmap.png")
print("- bar_per_class_accuracy_enhanced.png")
print("- roc_curves_enhanced.png")
print("- scatter_confidence_enhanced.png")
print("- violin_confidence_distribution.png")
