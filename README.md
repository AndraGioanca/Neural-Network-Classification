# Neural Network for Classification

## Overview
This project focuses on solving a multi-class classification problem using a neural network. The model is trained on the Iris dataset, a widely used dataset in machine learning, consisting of 150 instances with four numerical features representing petal and sepal dimensions. The dataset is divided into three classes of equal distribution.

## Learning Algorithms
Two training algorithms were implemented:

### **Gradient Descent Method**
A fundamental optimization technique used for neural network training. It iteratively updates the weights in the opposite direction of the gradient of the loss function (Cross-Entropy) to minimize the objective function.

### **Stochastic Gradient Descent (SGD) Method**
A variation of the Gradient Descent method that optimizes network weights using randomly selected subsets of the training set, significantly reducing computational cost per iteration.

## Neural Network Configuration
- **Number of hidden layer nodes:** 10
- **Hidden layer activation function:** f(z) = z * tanh(z)
- **Objective function:** Cross-Entropy Loss

## Results
### **Gradient Descent Method**
- **Epochs:** 2000
- **Learning Rate:** 0.001
- Achieved ideal classification results with a confusion matrix showing only diagonal values.

### **Stochastic Gradient Descent Method**
- **Epochs:** 2000
- **Learning Rate:** 0.001
- Faster minimization of the objective function compared to standard Gradient Descent.

## Conclusion
Both methods efficiently trained the neural network due to the small dataset size (150 instances). The dataset was split 80/20 for training and testing, and both approaches achieved perfect classification results based on the confusion matrix analysis.

## Project Structure
```
/NeuralNetwork-Classification
│── crossEntropyLoss.m       # Computes cross-entropy loss and gradient
│── customActivation.m       # Implements a custom activation function
│── feedforward.m            # Forward propagation function
│── initializeNetwork.m      # Initializes network weights and biases
│── softmax.m                # Softmax function for classification
│── trainGD.m                # Training using Gradient Descent
│── trainSGD.m               # Training using Stochastic Gradient Descent
│── Iris.csv                 # Dataset used for classification
│── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Neural-Network-Classification.git
   ```
2. Open MATLAB and navigate to the project directory.
3. Run the training script using:
   ```matlab
   [W1, W2, b1, b2, lossHistory, gradHistory, timeHistory] = trainGD(X_train, y_train, W1, W2, b1, b2, epochs, learningRate);
   ```

## Usage
- Modify `initializeNetwork.m` to adjust network parameters.
- Change hyperparameters in `trainGD.m` or `trainSGD.m`.
- Use `feedforward.m` to test new data after training.
