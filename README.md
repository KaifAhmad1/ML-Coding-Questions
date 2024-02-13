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
