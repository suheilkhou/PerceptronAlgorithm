# Perceptron Classifier

This repository contains an implementation of the multi-class Perceptron learning algorithm in Python.  
The code was developed as part of the Machine Learning 1 course.

## About

The classifier uses the classic Perceptron algorithm and supports multi-class classification.

The input is a CSV file where:
- Each row is a sample.
- The last column contains the class label.
- The file should not contain a header.

The model is trained until convergence or a maximum number of iterations is reached.

## Requirements

- Python 3.x
- numpy
- pandas

## What’s Included

- `PerceptronClassifier.py`: Main script containing the classifier implementation, training logic, and CLI interface.
- `iris_sep.csv`: A data sample that is linearly separable.

## How to Use

You can run the classifier using the terminal.

### With a test file:
```bash
python PerceptronClassifier.py iris_sep.csv test.csv
```
### Without a test file:
```bash
python PerceptronClassifier.py iris_sep.csv
# You will be prompted to enter a test set percentage (e.g., 0.2)
```

## Authors
The implementation was done by Suheil Khourieh.
