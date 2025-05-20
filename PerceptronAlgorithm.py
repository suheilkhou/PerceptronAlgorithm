import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self, max_iter: int = 10000):
        """
        Constructor for the PerceptronClassifier.

        :param max_iter: Maximum number of iterations for training. Default is 10000.
        """
        self.max_iter = max_iter
        self.W_ = None
        self.classes_ = None
        self.num_samples_ = 0
        self.num_features_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
                   Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. It is guaranteed to match X's rows in length.
                   Array datatype is guaranteed to be np.uint8.
        :return: True if a fit was found (the data is seperable) after max_iter iterations, false other wise.
        """
        self.num_samples_, self.num_features_ = X.shape  # Find the dimensions of the training data.
        # Add the value 1 for each point to deal with the bias.
        ones = np.ones((self.num_samples_, 1), dtype=X.dtype)
        X = np.append(X, ones, axis=1)
        self.classes_ = np.unique(y)  # The distinct classes (labels) of the model.
        self.K_ = len(self.classes_)  # The number of classes (labels).
        self.W_ = np.zeros((self.K_, self.num_features_ + 1), dtype=X.dtype)  # Initialize a matrix for the weights of the model.
        # Assuming the data is linearly separable, loop until no points are misclassified.
        for epoch in range(self.max_iter):
            flag = True
            for i in range(self.num_samples_):
                if self.check_prediction(X[i], y[i]):
                    continue
                flag = False
                predicted_index = self.predict_label_index(x=X[i])  # The index of the predicted label.
                correct_index = np.where(self.classes_ == y[i])[0][0]  # The index of the correct label.
                self.W_[correct_index] += X[i]    # Update the weight vector of the correct label by adding the misclassified point values.
                self.W_[predicted_index] -= X[i]  # Update the weight vector of the predicted label by removing the misclassified point values.
            if flag:
                break
        return flag

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
                   Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        if self.W_ is None or self.classes_ is None:
            raise ValueError("Model must be trained using .fit() before calling .predict().")
        num_samples = X.shape[0]  # Determine the size of the test set.
        # Add the value 1 for each point to deal with the bias.
        ones = np.ones((num_samples, 1), dtype=X.dtype)
        X = np.append(X, ones, axis=1)
        predictions = np.zeros(num_samples, dtype=np.uint8)  # Initialize the predictions array.
        # Iterate over the test set, predict each point, and store the prediction at the appropriate index.
        for i in range(num_samples):
            predictions[i] = self.classes_[self.predict_label_index(x=X[i])]
        return predictions  # Return the predictions of the model.

    def predict_label_index(self, x: np.ndarray) -> int:
        """
        This method predicts the label of a point and returns its index in the labels array.
        The prediction is done by finding the maximum value resulting from multiplying
        the weights matrix with the feature vector.
        :param x: A 1-dimensional numpy array that holds the feature values.
        :return: The index of the predicted label.
        """
        return np.argmax(np.dot(self.W_, x))

    def check_prediction(self, x: np.ndarray, correctClass) -> bool:
        """
        This method checks if the predicted label matches the actual label.
        :param x: A 1-dimensional numpy array that holds the feature values.
        :param correctClass: The actual class/label.
        :return: True if the prediction is correct, False otherwise.
        """
        prediction = self.classes_[self.predict_label_index(x)]
        return prediction == correctClass

if __name__ == "__main__":
    print("NOTE: The input CSV file must not include a header row.")
    print("It is assumed that the last column contains the labels, which must be numeric (0, 1, ..., k).")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str)
    parser.add_argument('--test-csv', type=str, default=None)
    args = parser.parse_args()
    
    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Loading data from {args.csv}...")
    train_data = pd.read_csv(args.csv, header=None)
    X_full = train_data[train_data.columns[:-1]].values.astype(np.float32)
    y_full = pd.factorize(train_data[train_data.columns[-1]])[0].astype(np.uint8)

    if args.test_csv:
        print(f"Loading test data from {args.test_csv}...")
        test_data = pd.read_csv(args.test_csv, header=None)
        test_X = test_data[test_data.columns[:-1]].values.astype(np.float32)
        test_y = pd.factorize(test_data[test_data.columns[-1]])[0].astype(np.uint8)
        X = X_full
        y = y_full
    else:
        while True:
            print("No test set provided. Enter the percentage of how how to split the data")
            percentage = float(input("For example, 0.2 splits the data into 20% test and 80% train randomlly: "))
            if 0 < percentage < 1:
                break
            print("Percentage must be between 0 and 1 (exclusive)!")

        num_samples = X_full.shape[0]
        num_elements = int(num_samples * percentage)
        indices = np.random.choice(num_samples, size=num_elements, replace=False)
        mask = np.zeros(num_samples, dtype=bool)
        mask[indices] = True
        X = X_full[~mask]
        y = y_full[~mask]
        test_X = X_full[mask]
        test_y = y_full[mask]
        print(len(X))
        print(len(test_X))

    print("Fitting...")
    is_separable = model.fit(X, y)
    if not is_separable:
        raise ValueError("The data is not linearly separable within the given number of iterations.")
    print("Done")
    y_pred = model.predict(test_X)
    print("Done")
    accuracy = np.sum(y_pred == test_y.ravel()) / test_y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")

    print("*" * 20)