# Churn Modeling Project

## Dataset Overview

The dataset used in this project contains the following features:

- **RowNumber**: The row number of the dataset.
- **CustomerId**: Unique ID of the customer.
- **Surname**: Customer's surname.
- **CreditScore**: Credit score of the customer.
- **Geography**: Country of residence.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products the customer has with the bank.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Whether the customer has exited (1 = Yes, 0 = No).

## Installation

Ensure you have the following packages installed:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tensorflow (keras)

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow
```
## Project Structure
The repository contains the following files:

•Practical - 6.ipynb: The Jupyter Notebook file containing the implementation for churn prediction using a deep learning model.  
•dataset.csv: The dataset file containing the churn modeling data used for training and testing the model.  
## How to Run
To run the project, follow these steps:

1. **Clone the Repository**
First, clone the repository to your local machine using the command:

```bash
git clone https://github.com/ParthYuki/Churn_Modeling.git
```
2. **Navigate to the Project Directory**
After cloning the repository, navigate to the project directory:

```bash
cd ParthYuki
```

3. **Install Required Dependencies**
Install the required Python packages using the following command:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow
```
4. **Open the Jupyter Notebook**
Open the Churn_Modeling.ipynb notebook file in Jupyter or any compatible environment:

```bash
jupyter notebook Practical - 6.ipynb
```
5. **Run All Cells**
Execute all the cells in the notebook to preprocess the data, train the model, and evaluate its performance.

## Model Overview
The model in this project is a neural network built using Keras. It includes:

• **Input Layer**: Takes the customer features as input.  
• **Hidden Layers**: Dense layers with ReLU and LeakyReLU activations, along with Batch Normalization and Dropout layers for regularization.  
• **Output Layer**: A single neuron with a sigmoid activation for binary classification.  
• **Loss Function**: Binary cross-entropy is used since this is a binary classification task.  
• **Optimizer**: Adam optimizer is used for training the model.

## Evaluation Metrics
The model performance is evaluated based on:

• **Accuracy**: Proportion of correctly classified instances.  
• **Precision, Recall, F1-Score**: To assess the model's performance more comprehensively.
## Results
The model achieved the following performance on the test dataset:

• **Accuracy**: Approximately 0.84  
• **Precision, Recall, F1-Score**: Detailed metrics available in the classification report.  
## Files in This Repository
• **dataset.csv**: The dataset used for training and evaluating the model.  
• **Churn_Modeling.ipynb**: Jupyter Notebook containing the code for data processing, model training, and evaluation.
