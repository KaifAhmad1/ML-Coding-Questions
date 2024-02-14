# ML-Coding-Questions
## Classical ML:
W.I.P
## Deep Learning:
#### What are the components of a basic neural network architecture?
A basic neural network architecture comprises:
 1. **Input Layer:** Receives input data.
 2. **Hidden Layers:** Process input through weighted connections.
 3. **Activation Function:** Adds non-linearity to neuron outputs.
 4. **Weights and Biases:** Parameters adjusted during training.
 5. **Output Layer:** Produces final network output.
 6. **Loss Function:** Measures the difference between predicted and actual output.
 7. **Optimization Algorithm:** Updates weights to minimize loss.
 8. **Learning Rate:** Controls step size during optimization.

#### Can you explain the purpose of activation functions in neural networks, and could you describe a few different types of activation functions commonly used in deep learning with their use cases?
- Activation functions play a crucial role in neural networks by introducing non-linearity to the model, enabling it to learn complex patterns and relationships in the data. Without activation functions, neural networks would essentially reduce to a series of linear transformations, limiting their ability to approximate complex functions.
- Here are a few commonly used activation functions in deep learning:
 1. **Sigmoid Function:** The sigmoid function, also known as the logistic function, squashes the input to the range `[0, 1]`.
  - **Use Cases:** It's often used in the output layer of binary classification tasks, where the goal is to predict probabilities.

     `f(x) = {1} / {1 + e^{-x}}`
 ``` Python 
 import numpy as np 
def sigmoid(x):
  cal = 1 / (1 + np.exp(-x))
  return cal

# Input 
x = np.array([ -1.0, 0.0, 1.0, 2.0 ])
print(sigmoid(x))
```
 2. **Tanh Function:** The hyperbolic tangent function, often abbreviated as `tanh`, is another popular activation function used in neural networks. It has properties similar to the sigmoid function but maps input values to the range `(-1, 1)` instead of (0, 1). This can be useful for situations where you need output values that are `symmetric` around zero.
  - **Use Cases:** Tanh activation function is crucial in neural networks for modeling nonlinear relationships, stabilizing gradients in deep networks, capturing sequential dependencies in RNNs, controlling output range in image classification, enabling recurrent layer functionality in NLP, facilitating unsupervised learning in autoencoders, and ensuring proper output scaling in GANs.

     `f{x} = (e^{x} - e^{-x}) / (e^{x} + e^{-x))`
``` Python 
import numpy as np 
def tanh(x):
  cal = ((np.exp(x) - np.exp(-x)) / 
          (np.exp(x) + np.exp(-x)))
  return cal 
# Input 
x = np.array([1, 3, 2, 9, -.2, -0.07])
print(tanh(x))
```
 3. **ReLU Function:** ReLU stands for Rectified Linear Unit. It is an activation function commonly used in neural networks. The ReLU function is defined as it simply outputs the input x if x is positive, and zero otherwise. In other words, it `rectifies` negative values to zero, while leaving positive values unchanged.

    `f{x} = max(0, x)`
  - **Use Cases:**  ReLU (Rectified Linear Unit) is a fundamental activation function in deep learning, widely valued for its ability to combat gradient vanishing, enhance computational efficiency, promote sparsity, and facilitate the training of deep neural networks. Its simplicity and effectiveness make it a cornerstone in various applications, including image classification, natural language processing, and beyond.
``` Python 
import numpy as np 
def relu(x):
  return np.maximum(0, x)
# Input 
x = np.array([2, 1, 0.3, -0.7, -2, 6])
print(relu(x))
```
4. **Leaky ReLU Function:** Leaky ReLU (Rectified Linear Unit) is an activation function used in deep learning that introduces a small, non-zero slope for negative inputs, preventing neurons from becoming completely inactive. It addresses the `dying ReLU` problem by allowing a small gradient for negative values, improving the robustness and effectiveness of neural networks.
 - **Use Cases:** Leaky ReLU keeps gradients stable for smoother training and prevents neuron saturation by staying active across a wider range of inputs. It boosts model resilience to noise, promotes efficient activation, handles tricky data well, and supports stable GAN training.
   
   `f{x} = max(alpha * x, x)` where alpha is a very small positive constant
   ``` Python 
   import numpy as np
   def leaky_relu(x):
     alpha = 0.01
     return np.where(x > 0, x, alpha * x)
   # Input 
   x = np.array([2, 1, 0.3, -0.7, -2, 6])
    print(leaky_relu(x))
   ```
5. **ELU Function:** ELU (Exponential Linear Unit) is an activation function used in deep learning. It behaves like the identity function for positive inputs, directly returning the input value. However, for negative inputs, it applies a non-linear transformation using the exponential function, resulting in a smooth curve that asymptotically approaches -1. This transformation helps prevent "dead" neurons during training, ensuring effective learning in neural networks.
 - **Use Cases:**
   - 1. **Robustness:** ELU prevents "dead" neurons and handles negative inputs effectively.
   - 2. **Smooth Gradient:** Provides stable training with a smooth gradient.
   - 3. **Rich Representation:** Allows for richer representation learning.
   - 4. **Vanishing Gradient:** Addresses the vanishing gradient problem effectively.
   - 5. **Implicit Regularization:** Acts as implicit regularization to prevent overfitting.


` f(x) = {  x  if x > 0, alpha * (e^x - 1)  if x <= 0} `
``` Python 
import numpy as np 
def elu(x):
  alpha = 0.01
  return np.where(x > 0, x, alpha * (np.exp(x) - 1))
# Input 
x = np.array([2, 1, 0.3, -0.7, -2, 6])
print(elu(x))
```
6. **Softmax Function:** The softmax activation function is commonly used, especially in the output layer of neural networks for multi-class classification tasks. It takes a vector of arbitrary real-valued scores (often called logits) and transforms them into a probability distribution over multiple classes.
   - **Use Cases:** The softmax activation function is commonly used in the output layer of neural networks for multi-class classification tasks. Its primary purpose is to generate a probability distribution over multiple classes, allowing the model to predict the most likely class for a given input. This function is widely employed in applications such as image classification, natural language processing, speech recognition, medical diagnosis, and gesture recognition.

     `softmax(z_i) = exp(z_i) / sum(exp(z_j) for j in range(K))`

   -  `softmax(z_i)`: Probability of the i-th class after softmax
   - `exp(z_i)`: Exponential of the i-th logit
   - `sum(exp(z_j) for j in range(K))`: Sum of exponentiated logits for all classes
   - `for j in range(K)`: Loop over all classes, where K is the total number of classes

``` Python 
import numpy as np
def softmax(z):
    e_z = np.exp(z - np.max(z))  # Subtracting the maximum value for numerical stability
    return e_z / np.sum(e_z)

# Input:
z = np.array([1.0, 2.0, 3.0])
softmax_output = softmax(z)
print("Softmax output:", softmax_output)
```
#### What is the purpose of optimization algorithms in deep learning? Can you explain some of the optimization algorithms in deep learning with their implementation and use cases? 
The purpose of optimization algorithms in deep learning is to minimize the loss function, improving the model's ability to make accurate predictions by adjusting its parameters iteratively during training.

There are several types of optimization algorithms used in deep learning, and they can be categorized based on whether they use derivative information and whether the objective function is differentiable. Here are some of the main types of optimization algorithms used in deep learning: 
 #### Gradient Descent:
 Gradient Descent is an optimization algorithm used to minimize the loss function in machine learning models. It iteratively adjusts the parameters of the model in the direction of the steepest descent of the gradient of the loss function. Here's a detailed explanation along with the algorithm and implementation
   - Initialize the parameters `theta` randomly or with some `predefined values`.
   - Repeat until convergence:
   - Compute the gradient of the loss function concerning the parameters: `gradient = compute_gradient(loss_function, theta)`.
   - Update the parameters using the gradient and the learning rate: `theta = theta - learning_rate * gradient`.
   - Check for convergence criteria (e.g., small change in the loss function or maximum number of iterations reached).

``` Python 
import numpy as np
# Gradient Descent optimization algorithm
def gradient_descent(loss_function, initial_theta, learning_rate, max_iterations=1000, epsilon=1e-6):
    theta = initial_theta  # Initialize parameters
    loss_values = []  # Track loss values
    
    for iteration in range(max_iterations):
        gradient = compute_gradient(loss_function, theta)  # Compute gradient
        theta -= learning_rate * gradient  # Update parameters
        loss = loss_function(theta)  # Compute loss
        loss_values.append(loss)  # Store loss value
        if len(loss_values) > 1 and abs(loss_values[-1] - loss_values[-2]) < epsilon:  # Check convergence
            break
    
    return theta, loss_values

# Compute gradient of the loss function
def compute_gradient(loss_function, theta, epsilon=1e-6):
    gradient = np.zeros_like(theta)  # Initialize gradient vector
    
    for i in range(len(theta)):
        theta_plus = theta.copy()  # Make a copy of theta
        theta_plus[i] += epsilon  # Perturb theta slightly
        gradient[i] = (loss_function(theta_plus) - loss_function(theta)) / epsilon  # Compute finite difference
    
    return gradient

# loss function (squared loss)
def squared_loss(theta):
    return (theta - 5) ** 2

# Set initial parameters and hyperparameters
initial_theta = np.array([0.0])
learning_rate = 0.3

# Run gradient descent optimization
optimized_theta, loss_values = gradient_descent(squared_loss, initial_theta, learning_rate)
print("Optimized theta:", optimized_theta)
print("Final loss:", loss_values[-1])
```

- Different flavours of Gradient Descent:
 1. **Batch Gradient Descent:** Batch Gradient Descent is an optimization algorithm used in machine learning where, during each iteration, it calculates the gradient of the cost function using the entire training dataset to update the model parameters in the direction that minimizes the cost.
  ``` Python 
  import numpy as np
  def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000, epsilon=1e-6):
    theta = np.zeros(X.shape[1])  # Initialize coefficients with zeros
    for _ in range(n_iterations):  # Perform gradient descent iterations
        gradient = X.T.dot(X.dot(theta) - y) / len(X)  # Compute gradient
        theta -= learning_rate * gradient  # Update coefficients
        if np.linalg.norm(learning_rate * gradient) < epsilon:  # Check convergence
            break
    return theta
  # Input:
  np.random.seed(0)
  X = 2 * np.random.rand(100, 3)
  y = 4 + np.dot(X, np.array([3, 5, 2])) + np.random.randn(100)
  X_b = np.c_[np.ones((100, 1)), X]
  theta = batch_gradient_descent(X_b, y)
  print("Optimized Coefficients:", theta)
  ```
  2. **Stochastic Gradient Descent:** Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest decrease of the function. At each iteration, the algorithm calculates the gradient of the loss function concerning the parameters (weights) of the model. It then updates the parameters by moving them in the opposite direction of the gradient, scaled by a small factor known as the learning rate.
     - In standard Gradient Descent, the algorithm computes the gradient of the loss function over the entire dataset for each iteration, which can be computationally expensive, especially for large datasets.
Stochastic Gradient Descent addresses this issue by updating the parameters based on the gradient of the loss function computed on a single random data point (or a small batch of data points) at each iteration.
This randomness in selecting a single data point or a batch makes the optimization process stochastic.
``` Python 
import numpy as np
def stochastic_gradient_descent(loss_function, initial_theta, learning_rate, dataset, max_epochs=100, epsilon=1e-6):
    theta = initial_theta
    loss_values = []
    
    for epoch in range(max_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0.0
        
        for data_point in dataset:
            gradient = compute_gradient(loss_function, theta, data_point)
            theta -= learning_rate * gradient
            epoch_loss += loss_function(theta, data_point)
        
        epoch_loss /= len(dataset)
        loss_values.append(epoch_loss)
        
        if len(loss_values) > 1 and abs(loss_values[-1] - loss_values[-2]) < epsilon:
            break
    
    return theta, loss_values

def compute_gradient(loss_function, theta, data_point, epsilon=1e-6):
    gradient = np.zeros_like(theta)
    
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        gradient[i] = (loss_function(theta_plus, data_point) - loss_function(theta, data_point)) / epsilon
    
    return gradient

def squared_loss(theta, data_point):
    x, y = data_point
    return (theta * x - y) ** 2

np.random.seed(0)
dataset_size = 100
X = np.random.rand(dataset_size)
y = 5 * X + np.random.randn(dataset_size)
dataset = np.vstack((X, y)).T

initial_theta = np.array([0.0])
learning_rate = 0.01

optimized_theta, loss_values = stochastic_gradient_descent(squared_loss, initial_theta, learning_rate, dataset)
print("Optimized theta:", optimized_theta)
print("Final loss:", loss_values[-1])
```
 3.  **Mini-batch Gradient Descent:** Mini-batch Gradient Descent is a variation of the Gradient Descent optimization algorithm. While standard Gradient Descent computes the gradient of the loss function over the entire dataset in each iteration (batch gradient descent) and Stochastic Gradient Descent computes the gradient based on a single random data point, Mini-batch Gradient Descent computes the gradient on small random batches of data points.
     - By using mini-batches, Mini-batch Gradient Descent combines the advantages of both batch and stochastic gradient descent. It reduces the computational burden compared to batch gradient descent while achieving faster convergence than stochastic gradient descent.
     - Typically, the size of the mini-batch is chosen based on computational resources and empirical performance. Common choices include 32, 64, or 128 data points per mini-batch.
``` Python 
import numpy as np

def mini_batch_gradient_descent(loss_function, initial_theta, learning_rate, dataset, batch_size=32, max_epochs=100, epsilon=1e-6):
    theta = initial_theta
    loss_values = []
    
    for epoch in range(max_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0.0
        
        for batch_start in range(0, len(dataset), batch_size):
            batch = dataset[batch_start:batch_start+batch_size]
            gradient = compute_gradient(loss_function, theta, batch)
            theta -= learning_rate * gradient
            epoch_loss += loss_function(theta, batch)
        
        epoch_loss /= len(dataset)
        loss_values.append(epoch_loss)
        
        if len(loss_values) > 1 and abs(loss_values[-1] - loss_values[-2]) < epsilon:
            break
    
    return theta, loss_values

def compute_gradient(loss_function, theta, batch, epsilon=1e-6):
    gradient = np.zeros_like(theta)
    
    for data_point in batch:
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            gradient[i] += (loss_function(theta_plus, data_point) - loss_function(theta, data_point)) / epsilon
    
    return gradient / len(batch)

def squared_loss(theta, batch):
    X = batch[:, 0]
    y = batch[:, 1]
    return np.mean((theta * X - y) ** 2)

np.random.seed(0)
dataset_size = 100
X = np.random.rand(dataset_size)
y = 5 * X + np.random.randn(dataset_size)
dataset = np.vstack((X, y)).T

initial_theta = np.array([0.0])
learning_rate = 0.01
batch_size = 10

optimized_theta, loss_values = mini_batch_gradient_descent(squared_loss, initial_theta, learning_rate, dataset, batch_size)
print("Optimized theta:", optimized_theta)
print("Final loss:", loss_values[-1])
```
