🌾 Rice-Weed Classification using MobileNetV2
This image can be changed to a visually appealing project image or a representative screenshot.

📝 Abstract
This project implements a deep learning solution for rice-weed classification using transfer learning with the MobileNetV2 architecture. The system achieved an exceptional test accuracy of 

95.44% on a dataset of 3,632 images across 11 distinct classes. This demonstrates the model's effective performance for precision agriculture applications by successfully distinguishing between rice plants and various weed species.


✨ Key Features

High Accuracy: The model shows excellent performance with a final test accuracy of 95.44%.


Efficient Architecture: The MobileNetV2 architecture is efficient and suitable for mobile and embedded devices.


Fast Training: The transfer learning approach significantly reduces the training time while maintaining high accuracy.


Robust Generalization: The model shows stable performance across training epochs and generalizes well to unseen data.



Confident Predictions: The model was able to classify new images with a very high confidence score of 92.21%.

📊 Performance and Architecture
Dataset Information
The dataset used for this project consists of rice and weed images from a public source. You can access the dataset here: Rice and Weeds Image Dataset.

The model was trained on a dataset of 3,632 images  with the following distribution:

Dataset Split	Number of Images
Training Set	2901
Validation Set	358
Test Set	373
Total Classes	11
Training Results
The model was trained for 10 epochs and showed rapid improvement in accuracy.

Metric	Result
Final Training Accuracy	
96.07% 

Final Validation Accuracy	
95.81% 

Final Test Accuracy		
95.44% 

Test Loss	
0.1191 

📂 Project Structure
train_model.py: Script to train the deep learning model.

evaluate_model.py: Script to evaluate the trained model's performance on the test set.

predict.py: Script to make predictions on new images.

split_dataset.py: Utility to split the dataset into training, validation, and test sets.

rice_weed_classifier_model.h5: The trained MobileNetV2 model file.

confusion_matrix_normalized.png: Visual representation of the model's classification performance.

roc_curves.png: Receiver Operating Characteristic curves for multi-class classification.

bar_per_class_accuracy.png: Bar chart showing the accuracy for each class.

scatter_confidence.png: Scatter plot of prediction confidence.

🚀 Getting Started
Prerequisites
Python 3.13 or newer 

Deep Learning Framework: TensorFlow/Keras 

Other dependencies can be installed using a requirements.txt file.

Installation
Clone this repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install the required Python libraries.

Usage
Making a Prediction
Use the predict.py script to classify a new image. The model correctly classified a test image named 

leaf1.jpg as W_CL_11_Synedrella nodiflora with 92.21% confidence in a validation experiment.

Bash

python predict.py --image_path path/to/your/image.jpg
Evaluating the Model
To evaluate the model on the test set, run the evaluate_model.py script.

Bash

python evaluate_model.py
📈 Future Work

Real-time Deployment: Implement the model on mobile devices for field use by farmers.


Drone Integration: Integrate the system with drone-based imaging for large-scale agricultural monitoring.


Object Detection: Implement object detection techniques for precise localization of weeds within an image.


Expand Dataset: Add more regional weed species and crops to the dataset to improve generalization.

👨‍💻 Author

Devansh Bansal Student, VIT Bhopal University 