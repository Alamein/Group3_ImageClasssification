# Group3_ImageClasssification

# Nigerian Food Classification Project Documentation

## Introduction
The Nigerian Food Classification project aims to develop a deep learning model using PyTorch to classify various Nigerian food dishes. The project utilizes a dataset sourced from Kaggle containing images of different Nigerian foods, including Jollof Rice, Pounded Yam, Egusi Soup, Suya, and others. This documentation provides an overview of the project, including its objectives, dataset, model architecture, training process, and usage instructions.

## Objectives
- Develop an image classification model capable of accurately identifying various Nigerian food dishes.
- Utilize deep learning techniques, specifically convolutional neural networks (CNNs), implemented using PyTorch.
- Train the model on a dataset consisting of images of Nigerian foods sourced from Kaggle.
- Evaluate the model's performance using appropriate metrics and techniques.

## Dataset
The dataset used in this project consists of images of various Nigerian food dishes collected from Kaggle. The dataset contains a total of 1492 images belonging to 18 classes, including:
- Jollof Rice
- Pounded Yam
- Egusi Soup
- Suya
- Other Nigerian food dishes

The images are organized into a directory structure where each class has its folder containing the respective images.

## Result and Discussion
Our research shows the performance of MobileNetV2 model on indigenous Nigeria food
image classification achieved the best performance performance with an prediction
score of 80%. However,we have not explored some others image classification models
such as AlexNet, VGGNet,ResNet etc to compare their performance on the Nigeria food
image classification [Link to the video Presentation:]( https://drive.google.com/file/d/16W9fqprLG5669TOFarQ0SzbE13YbcCCl/view?usp=drive_link)

## Recommendation 
In this research study, we proposed a framework for Indigenous Nigeria food image
classification. An accuracy of 80% was achieved on the second iteration which was an
improvement on the 73%, 80% which was recorded at every iterations respectively. The
results obtained revealed that the framework is capable of producing state-of the-art
results due to a high level of accuracy of the classifications obtained given the relatively
limited training dataset used. <br> 

Further work should dig further on food image classification according to Nigeria
regions such as North, East, south and west types of foods

## Team Members
 Bala Mairiga Abduljalil | ballaabduljalil@gmail.com <br>
 Aminu Hamza Nababa | alaminhnab4@gmail.com <br>
 Al-Amin Musa Magaga  | alaminmusamagaga@gmail.com <br> 
 Lurwanu Abdullahi  | lurwanabdullahi2107@gmail.com <br> 
<!--## Model Architecture
The image classification model is based on a convolutional neural network architecture implemented using PyTorch. The model architecture consists of the following components:
- Input Layer: Accepts input images of size (3, 224, 224) corresponding to RGB images resized to 224x224 pixels.
- Pre-trained CNN Backbone: Utilizes a pre-trained CNN backbone (e.g., ResNet, VGG, etc.) to extract features from input images.
- Fully Connected Layers: Additional fully connected layers are added to the model for classification purposes.
- Output Layer: Produces output probabilities for each class using softmax activation.

## Training Process
The training process involves the following steps:
1. Data Loading: Load the dataset using PyTorch's DataLoader, applying necessary data augmentation and preprocessing techniques.
2. Model Initialization: Initialize the CNN model architecture, optionally loading pre-trained weights.
3. Model Training: Train the model on the training dataset using techniques such as mini-batch gradient descent and backpropagation.
4. Model Evaluation: Evaluate the trained model's performance on the validation dataset, monitoring metrics such as accuracy, precision, recall, and F1 score.
5. Model Fine-tuning: Fine-tune the model as necessary based on performance evaluation results, adjusting hyperparameters and architecture if needed.
6. Model Saving: Save the trained model weights for future use and deployment.

## Usage
To use the Nigerian Food Classification model, follow these steps:
1. Clone the project repository from GitHub or download the project files locally.
2. Ensure that PyTorch and other dependencies are installed on your system.
3. Preprocess and organize your Nigerian food dataset into the required directory structure.
4. Train the model using the provided training script or notebook, specifying the dataset path, hyperparameters, and other configurations.
5. Evaluate the trained model's performance using the evaluation script or notebook, providing the path to the validation dataset.
6. Fine-tune the model based on performance evaluation results, adjusting hyperparameters and architecture if necessary.
7. Once satisfied with the model's performance, deploy it for inference on new images of Nigerian food dishes.

<!---## Conclusion
The Nigerian Food Classification project demonstrates the application of deep learning techniques using PyTorch for image classification tasks. By leveraging a dataset of Nigerian food images sourced from Kaggle, the project aims to develop an accurate and robust model capable of identifying various Nigerian food dishes. The provided documentation outlines the project's objectives, dataset, model architecture, training process, and usage instructions, enabling users to replicate and extend the work for their applications.
