🌾 Rice-Weed Classification using MobileNetV2
📝 Abstract

This project presents a deep learning approach for rice-weed classification using MobileNetV2 with transfer learning. Trained on 3,632 images across 11 classes, the model achieved an impressive 95.44% test accuracy, demonstrating strong potential for precision agriculture by distinguishing rice plants from various weed species.

✨ Key Highlights

High Accuracy – Test accuracy of 95.44% with stable generalization.

Efficient & Lightweight – MobileNetV2 enables deployment on mobile/embedded devices.

Fast Training – Transfer learning reduces training time significantly.

Confident Predictions – Example: Classified leaf1.jpg with 92.21% confidence.

📊 Dataset & Results

Dataset Size: 3,632 images (11 classes)

Split: 2,901 train | 358 validation | 373 test

Performance:

Training Accuracy: 96.07%

Validation Accuracy: 95.81%

Test Accuracy: 95.44%

Test Loss: 0.1191

📂 Project Structure

train_model.py – Train the model

evaluate_model.py – Evaluate performance

predict.py – Predict on new images

split_dataset.py – Dataset preparation

rice_weed_classifier_model.h5 – Trained model

Visualization outputs: confusion matrix, ROC curves, per-class accuracy, confidence plots

🚀 Usage

Clone the repository:

git clone https://github.com/Devansh-Bansal-AI/Rice-Weed-Classifier
cd Rice-Weed-Classifier
pip install -r requirements.txt


Run prediction:

python predict.py --image_path path/to/image.jpg


Evaluate model:

python evaluate_model.py

📈 Future Scope

Real-time deployment on mobile devices

Drone integration for large-scale monitoring

Object detection for precise weed localization

Expanded datasets with regional weed species

👨‍💻 Author

Devansh Bansal – Student, VIT Bhopal University
