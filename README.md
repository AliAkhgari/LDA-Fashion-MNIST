# LDA-Fashion-MNIST
Implementation of Linear Discriminant Analysis (LDA) from scratch.

## Introduction
Linear Discriminant Analysis (LDA) is a dimensionality reduction and classification technique used in machine learning and statistics. It aims to find a combination of features (or variables) that best separates or discriminates between different classes or groups within a dataset. LDA maximizes the ratio of the between-class variance to the within-class variance, making it effective for feature extraction and classification tasks. It is commonly used in pattern recognition, face recognition, and various classification problems to improve model performance by reducing the dimensionality of the data while preserving class-specific information.

## Usage

```python
import pandas as pd
from LDA import LDA

x_train = pd.read_csv("Fashion-MNIST/trainData.csv")
x_test = pd.read_csv("Fashion-MNIST/testData.csv")

y_train = pd.read_csv("Fashion-MNIST/trainLabels.csv")
y_test = pd.read_csv("Fashion-MNIST/testLabels.csv")

lda = LDA(x=x_train, y=y_train)

separability_trace = lda.separability_trace()
