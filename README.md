# ML-Coding-Questions
### ML Interview Questions(Theory):

- ### What is machine learning?
- Machine learning is a field of artificial intelligence where systems are trained to learn from data to make predictions or decisions without being explicitly programmed. Essentially, it's teaching machines to recognize patterns and make decisions based on that data.


- ### Explain the difference between supervised and unsupervised learning.
-  **Supervised learning** involves training a model on labeled data, where the input data is paired with the correct output. The model learns to predict the output from the input during training.
-  **Unsupervised learning** involves training a model on unlabeled data, where the model tries to learn the patterns and structures inherent in the data without explicit guidance on the output.

- ### What is overfitting, and how do you prevent it?
- Overfitting occurs when a machine learning model learns the training data too well, capturing noise or random fluctuations in the data rather than the underlying patterns. To prevent overfitting, techniques such as regularization, cross-validation, and using more data can be employed.

- ### Explain Bias-Variance Tradeoff?
- The bias-variance tradeoff is a fundamental concept in machine learning that deals with the balance between the error introduced by bias (the simplifying assumptions made by the model) and the error introduced by variance (the model's sensitivity to small fluctuations in the training data).

- ### Explain Cross-Validation in detail.
- Cross-validation is a way to check how well a machine-learning model will perform on new data. Here's the scoop:
- **What is it?** It's like a test for your model. Instead of just using one chunk of data to train and test, you split your data into smaller chunks called "folds". Then you train your model on some folds and test it on others, switching it up each time.
- **Why does it matter?** It's super important because it helps prevent your model from getting too good at memorizing the training data (overfitting) or not learning enough from it (underfitting). By trying out different chunks of data, you get a better sense of how your model will handle new, unseen data.
 - **Different flavours:**

- 1. **K-Fold Cross-Validation:** Split your data into k equal parts. Train the model on k-1 parts and test it on the remaining part. Repeat this k times, each time using a different part as the test set. Then average the results.
- 2. **Leave-One-Out Cross-Validation (LOOCV):** Take one data point out as a test set, train the model on the rest, and repeat for each data point. It's like K-Fold where k equals the number of data points.
- 3. **Stratified K-Fold Cross-Validation:** Similar to K-Fold, but it ensures that each fold has a similar distribution of target classes. Good for imbalanced datasets.
- 4. **Time Series Cross-Validation:** Important for data that has a time-based sequence. It keeps the time order intact when splitting the data into folds.

- ### What is regularization, and why is it important in machine learning?
- Regularization in machine learning is a technique used to prevent overfitting and improve the generalization of a model. Overfitting occurs when a model learns to perform well on the training data but fails to generalize to new, unseen data. Regularization adds a penalty term to the model's objective function, discouraging overly complex models that may fit the noise in the training data too closely.
- There are different types of regularization techniques, but two common ones are L1 and L2 regularization:
- **L1 Regularization (Lasso Regression):**
- 1. L1 regularization adds a penalty term proportional to the absolute value of the coefficients of the model.
- 2. It encourages sparsity in the model, meaning it tends to force irrelevant features' coefficients to be zero.
- 3. L1 regularization can be useful for feature selection as it effectively eliminates irrelevant features from the model.
   
- **L2 Regularization (Ridge Regression):**
- 1. L2 regularization adds a penalty term proportional to the squared magnitude of the coefficients of the model.
- 2. It tends to shrink the coefficients of the model towards zero, but rarely exactly to zero.
- 3. L2 regularization is good at reducing the magnitude of the coefficients without necessarily eliminating them, thus making it less aggressive in feature selection compared to L1 regularization.
- Other than L1 and L2:
- **Elastic Net Regularization:** This is a combination of L1 and L2 regularization. It adds both the L1 and L2 penalty terms to the loss function, with a hyperparameter that controls the balance between the two.
- **Dropout:** Commonly used in neural networks, dropout regularization randomly sets a fraction of the input units to zero during training. This helps prevent overfitting by reducing the reliance on specific neurons.
- **Early Stopping:** Instead of adding a penalty term to the loss function, early stopping stops the training process when the performance on a validation set starts to degrade, thereby preventing overfitting.
- **Batch Normalization:** This involves normalizing the inputs of each layer in a neural network to have zero mean and unit variance. This can act as a form of regularization and speed up training.

- **Explain the difference between classification and regression:**
- Classification categorizes data into predefined classes or categories, producing discrete outputs, while regression predicts continuous numerical values, providing outputs that fall along a spectrum.

- ### What is regression analysis?
- Regression analysis is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It aims to predict the value of the dependent variable based on the values of the independent variables.
- ### What are the different types of regression analysis?
- Common types include:
- 1.  linear regression
- 2.  polynomial regression
- 3.  logistic regression
-  4.  ridge regression
-  5.  lasso regression
-  6.  elastic net regression among others.
- ### What is  linear regression:
-  linear regression is a method to model the relationship between two continuous variables, typically denoted as 
**Linear Regression**

Linear regression is a statistical method used to model the relationship between two continuous variables. It is commonly denoted as:

$$Y = \beta_0 + \beta_1X + \varepsilon$$

**Formula:**
- $$\**Y:** Dependent variable$$
- **X:** Independent variable
- **$$\beta_0$$:** Y-intercept
- **$$\beta_1$$:** Slope of the line
- **$$\varepsilon$$:** Error term


  



