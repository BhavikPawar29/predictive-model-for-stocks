import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from randomForest import RandomForest
from files import file_paths
from expSmoothed import process_all_files

# Parameters
alpha = 0.3  # Smoothing factor
d = 2  # Number of days for prediction
n = 1 # Number of files OR len(file_paths) - To Process all available files

# Process n files
X, y = process_all_files(file_paths, alpha, d, n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

rf = RandomForest()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc}")

# Generate classification report
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)

# Plot OOB error
rf.plot_oob_error()
