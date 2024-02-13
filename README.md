# ML-Coding-Questions
### Classical ML:
#### Implement Linear Regression from Scratch? 
- Class `LinearRegression` with `fit` and `predict` methods.
- fit calculates the `slope` and intercept of the regression line using input data.
- predict predicts output values using the trained model.
- In the __main__ block, sample data is used to train the model, and a test data point is used to make a prediction.
``` Python 
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            # Predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Input:
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([3, 6, 9])

model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print(predictions)

```

#### Implement Logistic Regression from Scratch. 
- Defines a `LogisticRegression` class for binary classification.
- Utilizes NumPy for numerical computations.
- Constructor initializes `learning rate` and number of `iterations.`
- fit method trains the model using logistic regression.
- predict method predicts class labels based on the trained model.
- Example usage trains the model with sample data and makes predictions on test data.

``` Python 
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Sigmoid function
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

# Input:
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print(predictions)

```

### Implement the K-nearest neighbours algorithm to classify data points into different classes.
- **Training Phase:**
   1. Store all training samples and their corresponding class labels.
- **Prediction Phase:**
   1. Calculate distances to all training samples.
   2. Select the K nearest neighbours based on distance.
   3. Determine the majority class among the K neighbours.
   4. Assign this class to the test sample.

``` Python
from collections import Counter
import numpy as np

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for x_test in X_test:
        distances = [np.linalg.norm(x_train - x_test) for x_train in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [y_train[i] for i in nearest_indices]
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    return predictions

# Input:
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[5, 6], [0, 1]])

predictions = knn_predict(X_train, y_train, X_test, k=3)
print("Predictions:", predictions)
```

### Implement PCA in Python to reduce the dimensionality of a dataset.
-  PCA is a dimensionality reduction technique used to simplify complex datasets by reducing the number of features while preserving the most important information.
- **Process:**
  1. PCA identifies a new set of orthogonal axes called principal components that capture the maximum variance in the data.
  2. These principal components are linear combinations of the original features.
  3. The first principal component explains the most variance in the data, followed by the second, third, and so on.
- **Steps:**
  1. **Mean Centering:** Subtract the mean from each feature to centre the data around the origin.
  2. **Covariance Matrix:** Compute the covariance matrix of the mean-centered data.
  3. **Eigenvectors and Eigenvalues:** Calculate the eigenvectors and eigenvalues of the covariance matrix.
  4. **Select Components:** Sort the eigenvectors based on eigenvalues in descending order and select the top n_components.
  5. **Projection:** Project the original data onto the selected principal components to obtain the transformed dataset.
- **Assumptions:**
  - PCA assumes that the data is linearly correlated and that the principal components capture the most significant directions of variation.
``` Python 
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Compute mean of the data
        self.mean = np.mean(X, axis=0)
        # Center the data
        X = X - self.mean
        # Compute covariance matrix
        cov_matrix = np.cov(X.T)
        # Compute eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvectors based on eigenvalues and select top n_components
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]

    def transform(self, X):
        # Center the data
        X = X - self.mean
        # Project the data onto the principal components
        return np.dot(X, self.components)

# Example usage:
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

pca = PCA(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)

print("Original data:\n", X)
print("Transformed data:\n", X_transformed)
```

### Create a basic neural network with one hidden layer and implement forward and backward propagation for training on a simple dataset.
 - Initialize the neural network with parameters (input size, hidden layer size, output size, learning rate) and random weights/biases.
- **Forward Propagation:**
   1. Compute hidden layer input by multiplying input data with weights and adding bias.
   2. Apply activation function (e.g., sigmoid) to get hidden layer output.
   3. Compute output layer input by multiplying hidden layer output with weights and adding bias.
   4. Obtain final output of the neural network.
- **Backward Propagation:**
   1. Compute error between predicted and actual output.
   2. Compute gradients of error with respect to weights and biases.
   3. Propagate error back to hidden layer.
Update weights and biases using gradients and learning rate.
Iterate over training data for fixed number of epochs, performing forward and backward propagation.
After training, use the neural network for prediction by performing forward propagation on new data.

``` Python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, X, y):
        # Backward propagation
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_hidden_output += self.learning_rate * np.sum(output_delta, axis=0)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_input_hidden += self.learning_rate * np.sum(hidden_delta, axis=0)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)

# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=10000)

# Test the trained network
print("Predictions after training:")
print(nn.forward(X))

```








### NLP Engineer 
Tokenization: Implement a function to tokenize a given text.
Stopwords Removal: Write a program to remove stopwords from a text.
Stemming: Implement a stemming algorithm such as Porter or Snowball.
Lemmatization: Create a function to perform lemmatization on a text.
Bag of Words: Implement the bag of words model for a given corpus.
TF-IDF: Write a program to calculate TF-IDF scores for words in a corpus.
N-grams: Implement a function to generate n-grams from a text.
Word Frequency: Write a program to calculate the frequency of each word in a text.
Sentence Segmentation: Create a function to split a text into sentences.
Named Entity Recognition (NER): Implement a basic NER system.
Part-of-Speech (POS) Tagging: Write a program to perform POS tagging on a text.
Word Embeddings: Implement Word2Vec or GloVe for generating word embeddings.
Text Similarity: Write a program to calculate similarity between two texts.
Sentiment Analysis: Implement a sentiment analysis classifier.
Topic Modeling: Create a program to perform topic modeling using techniques like LDA.
Named Entity Linking (NEL): Implement a basic NEL system.
Dependency Parsing: Write a program to perform dependency parsing on a sentence.
Named Entity Disambiguation (NED): Implement a basic NED system.
Coreference Resolution: Create a function to resolve coreferences in a text.
Text Classification: Implement a text classifier for a given set of categories.
Sequence Labeling: Write a program to perform sequence labeling tasks like named entity recognition.
Text Generation: Implement a text generation model using techniques like LSTM or Transformer.
Machine Translation: Create a program for translating text from one language to another.
Spell Checking: Write a spell checking program for a given text.
Text Summarization: Implement an extractive or abstractive text summarization algorithm.
Language Detection: Create a program to detect the language of a given text.
Text Normalization: Implement text normalization techniques like case folding and accent removal.
Semantic Role Labeling (SRL): Write a program to perform semantic role labeling on a sentence.
Relation Extraction: Implement a system to extract relations between entities in a text.
Coreference Resolution: Write a program to resolve coreferences in a text.
Question Answering: Create a question answering system using techniques like BERT.
Text Segmentation: Implement a program to segment a text into coherent parts.
Sentiment Analysis on Social Media: Write a sentiment analysis classifier specifically for social media text.
Named Entity Recognition for Biomedical Text: Implement a NER system tailored for biomedical texts.
Text Generation in Dialog Systems: Create a text generation model for use in dialog systems.
Text Classification with Deep Learning: Implement a deep learning-based text classifier.
Named Entity Recognition with Transfer Learning: Write a NER system using transfer learning techniques.
Sentiment Analysis with Transformers: Implement a sentiment analysis model using transformer architectures.
Text Summarization with Reinforcement Learning: Create a text summarization model using reinforcement learning techniques.
Machine Translation with Attention Mechanism: Write a machine translation model using attention mechanisms.
Named Entity Recognition for Social Media: Implement a NER system specifically tailored for social media texts.
Coreference Resolution with Neural Networks: Write a coreference resolution model using neural network architectures.
Text Classification with BERT: Implement a text classification model using BERT.
Text Generation with GPT-3: Create a text generation model using OpenAI's GPT-3.
Sentiment Analysis with LSTM: Write a sentiment analysis model using LSTM neural networks.
Named Entity Recognition with CRF: Implement a NER system using conditional random fields.
Text Summarization with Transformer: Create a text summarization model using transformer architectures.
Machine Translation with Sequence-to-Sequence Models: Write a machine translation model using sequence-to-sequence models.
Text Classification with CNN: Implement a text classification model using convolutional neural networks.
Named Entity Recognition with BiLSTM-CRF: Write a NER system using a combination of bidirectional LSTM and conditional random fields.

### Infmrmation Retrieval: 
Boolean Retrieval Model: Implement a simple Boolean retrieval model to retrieve documents based on Boolean queries.
TF-IDF Calculation: Write a program to calculate TF-IDF scores for terms in a given corpus.
Vector Space Model (VSM): Implement a basic VSM for document retrieval based on cosine similarity.
Inverted Index Construction: Create an inverted index from a collection of documents.
Term Frequency Calculation: Write a program to calculate the term frequency of terms in a document.
Document Frequency Calculation: Implement a function to calculate the document frequency of terms in a corpus.
Cosine Similarity Calculation: Write a program to calculate the cosine similarity between two documents.
Okapi BM25: Implement the Okapi BM25 algorithm for document ranking.
Language Models for Information Retrieval: Implement a basic language model for document retrieval.
Pagerank Algorithm: Write a program to compute the Pagerank of web pages in a network.
HITS Algorithm: Implement the HITS algorithm for authority and hub scores computation.
Index Compression Techniques: Implement index compression techniques like delta encoding or variable byte encoding.
Top-k Retrieval: Write a function to retrieve the top-k documents for a given query.
Term Weighting Schemes: Implement different term weighting schemes like binary, TF, IDF, etc., for document retrieval.
Latent Semantic Indexing (LSI): Implement LSI for document retrieval based on singular value decomposition.
PageRank with Damping Factor: Write a program to compute PageRank with a damping factor.
Inverted Index Compression: Implement compression techniques like front coding or gamma encoding for the inverted index.
Spelling Correction: Write a program to correct misspelled terms in a query using techniques like edit distance.
Document Clustering: Implement a document clustering algorithm for organizing search results.
Query Expansion: Implement query expansion techniques like pseudo-relevance feedback or synonym-based expansion.
k-Nearest Neighbors (k-NN) Search: Implement a k-NN search algorithm to find the nearest neighbors of a query vector.
Approximate Nearest Neighbors (ANN): Implement an algorithm for approximate nearest neighbor search, such as Locality Sensitive Hashing (LSH).
Inverted Index with Vector Space Model: Extend an inverted index to support vector space model operations like cosine similarity.
k-Means Clustering: Implement the k-means clustering algorithm for vector data.
Hierarchical Clustering: Write a program to perform hierarchical clustering on a set of vectors.
Vector Quantization: Implement vector quantization techniques like k-means clustering for codebook generation.
Binary Space Partitioning (BSP): Implement BSP trees for efficient spatial indexing of high-dimensional vectors.
Product Quantization: Implement product quantization for vector compression and efficient nearest neighbor search.
Locality-Sensitive Hashing (LSH): Implement LSH for approximate nearest neighbor search in high-dimensional spaces.
MinHashing: Write a program to perform MinHashing for estimating Jaccard similarity between sets of vectors.
Implement the construction phase of the HNSW algorithm. Given a dataset of high-dimensional vectors, write a program to build the HNSW graph structure.
