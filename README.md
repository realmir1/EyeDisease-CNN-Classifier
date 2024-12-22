# Eye Dataset Classification with TensorFlow and Keras 🚀

This project showcases the implementation of a Convolutional Neural Network (CNN) to classify eye images into predefined categories. The primary focus is to demonstrate the end-to-end workflow of image classification, starting from dataset preparation to training, evaluation, and visualization of results.

## Project Overview

Eye classification is an essential task in medical image analysis and computer vision. This project builds a deep learning model capable of identifying specific categories of eye images from a dataset. Using TensorFlow and Keras, the model leverages CNN layers to extract features, classify images, and deliver accurate predictions.

## Key Steps in the Project

### 1. Dataset Preparation 📊
The dataset is assumed to be organized into subfolders, each corresponding to a specific class. The project utilizes the following steps to prepare the dataset:
- **Image Resizing**: All images are resized to 200x200 pixels to ensure uniformity and compatibility with the model.
- **Data Normalization**: Pixel values are rescaled to the range [0, 1] for faster convergence during training.
- **Data Splitting**: The `ImageDataGenerator` class splits the dataset into training (90%) and validation (10%) subsets using its `validation_split` parameter.

### 2. Model Architecture 🏛️
The CNN model is designed using the `Sequential` API, which allows layers to be stacked in order. The architecture consists of:
- **Convolutional Layers**: Extract spatial features from input images using 32, 64, and 128 filters.
- **MaxPooling Layers**: Reduce the dimensions of feature maps, minimizing computational cost and retaining critical information.
- **Dropout Layer**: Introduced after dense layers to reduce overfitting by randomly disabling neurons during training.
- **Dense Layers**: Fully connected layers are added for classification. The final layer uses the `softmax` activation function to output class probabilities for the 4 categories.

### 3. Model Compilation ✅
The model is compiled with the following configurations:
- **Optimizer**: `Adam`, known for its efficient and adaptive learning rate.
- **Loss Function**: `categorical_crossentropy`, suitable for multi-class classification problems.
- **Metric**: `accuracy`, used to monitor the model's performance during training and validation.

### 4. Model Training 🏃🏽
The model is trained using the prepared training dataset for 10 epochs. During training:
- The model learns to recognize patterns in images using the convolutional layers.
- Validation data is used to monitor performance and prevent overfitting.

### 5. Evaluation and Visualization 👁️
The model's performance is evaluated using several methods:
- **Accuracy and Loss Graphs**: Training and validation accuracy and loss are plotted to visualize the model's learning curve.
- **Predictions**: Sample predictions are generated using the test dataset, displaying the true labels alongside the predicted labels.

### 6. Examples ❓
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*wwnExqe720PPHykHhs5Hqw.png" alt="Resim 1" width="300"/>
  <img src="https://studyopedia.com/wp-content/uploads/2022/12/Matplotlib-featured-image-studyopedia.png" alt="Resim 2" width="300"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/2560px-TensorFlow_logo.svg.png" alt="Resim 3" width="200"/>
</p>


## Visualizations
1. **Training Metrics**:
   - Accuracy and loss plots provide insights into the model's learning behavior over epochs.
   - Validation curves highlight the model's generalization ability to unseen data.
2. **Sample Predictions**:
   - A set of 5 images from the test dataset is visualized, showing the true and predicted class labels. This provides a clear understanding of the model's practical performance.

## How to Use This Project

1. **Setup Dataset**:
   Ensure the dataset is structured as follows:
   ```
   Eye dataset/
       ├── Class1/
       ├── Class2/
       ├── Class3/
       └── Class4/
   ```
   Replace `Class1`, `Class2`, etc., with actual class names.

2. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. **Update the Dataset Path**:
   Modify the `data_dir` variable in the code to point to your dataset's location.

4. **Run the Code**:
   Execute the Python script to train the model, generate evaluation plots, and display predictions.

## Results and Insights
- The model achieves promising accuracy, demonstrating its ability to classify eye images into multiple categories.
- Visualization of training metrics and sample predictions helps understand the model's strengths and areas for potential improvement.

## Future Work
- Experiment with advanced data augmentation techniques to improve model generalization.
- Explore deeper architectures or pre-trained models (e.g., VGG16, ResNet) for enhanced accuracy.
- Evaluate the model on additional datasets or real-world data for broader applicability.

This project provides a foundational understanding of building CNN models for image classification tasks, enabling further exploration and improvements in medical imaging applications.

## Library 📚
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*wwnExqe720PPHykHhs5Hqw.png" alt="Resim 1" height="125"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Matplotlib_icon.svg/1200px-Matplotlib_icon.svg.png" alt="Resim 2" height="125">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/2560px-TensorFlow_logo.svg.png" alt="Resim 3" height="125"/>
  <img src="https://logosandtypes.com/wp-content/uploads/2024/02/NumPy.png" alt="resim4" height="125"/>
</p>
