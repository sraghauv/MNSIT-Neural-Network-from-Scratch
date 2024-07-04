## MNIST Neural Network - Numpy Implementation

### Description
This project is an implementation of a neural network designed to classify the MNIST dataset using only Numpy. By avoiding high-level libraries such as TensorFlow or PyTorch, this project focuses on understanding the fundamental mechanics of neural networks. The highest accuracy achieved is 95.04% on the test set. The neural network structure includes one hidden layer with ReLU as the activation function and a softmax output layer.

### Features
- **Numpy-Based Implementation**: The neural network is built from scratch using Numpy, providing a clear view of the underlying computations.
- **Single Hidden Layer**: The network contains one hidden layer to keep the structure simple and educational.
- **Activation Functions**: 
  - **ReLU**: Used as the activation function for the hidden layer.
  - **Softmax**: Used as the activation function for the output layer, enabling classification across the 10 digits.
- **Accuracy**: Achieves a highest accuracy of 95.04% on the test set.

### Technologies Used
- **Numpy**: For matrix operations and numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-numpy-neural-network.git
   ```
2. Navigate to the project directory:
   ```bash
   cd mnist-numpy-neural-network
   ```
3. Ensure you have the required dependencies installed:
   ```bash
   pip install numpy pandas matplotlib
   ```
4. Run the main script:
   ```bash
   python main.py
   ```

### Neural Network Structure
- **Input Layer**: 784 neurons (one for each pixel in the 28x28 input images).
- **Hidden Layer**: User-defined number of neurons with ReLU activation.
- **Output Layer**: 10 neurons with Softmax activation for digit classification (0-9).

### Results
- **Highest Accuracy**: 95.04% on the test set.

### Functions Overview
- **data_initalization(m)**: Initializes the training data.
- **data_test_initalization(m)**: Initializes the test data.
- **one_hot_encode(labels)**: Encodes labels using one-hot encoding.
- **randomize_paramters(node_size, m)**: Randomizes weights and biases.
- **reLU(x)**: ReLU activation function.
- **delta_reLU(x)**: Derivative of the ReLU activation function.
- **soft_max(Z)**: Softmax function.
- **forward(w1, w2, b1, b2, x)**: Performs forward propagation.
- **back_prop(a1, a2, z1, w2, one_hot, m, x)**: Performs backpropagation.
- **implement_loss(w1, w2, b1, b2, dW2, dB2, dW1, dB1, learning_rate)**: Updates weights and biases.
- **terminal()**: Handles user inputs.
- **get_predictions(A2)**: Gets predictions from the neural network.
- **get_accuracy(predictions, Y)**: Calculates the accuracy of the predictions.
- **gradient_descent(x, one_hot, m, hidden_layer_size, iterations, learning_rate, labels)**: Performs gradient descent for training.
- **test(w1, w2, b1, b2, x, labels, m)**: Tests the trained model.
- **image_view(w1, w2, b1, b2, x_test, labels_test, m)**: Displays images and predictions.
- **main()**: Main function to execute the program.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact
For any inquiries or feedback, please contact [your email].

Feel free to explore the code, suggest improvements, and contribute to the project!
