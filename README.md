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

