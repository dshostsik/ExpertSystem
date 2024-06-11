import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from plots import plot_feature_importance, plot_decision_tree, plot_confusion_matrix_custom


def load_data(file_path):
    data = pd.read_csv(file_path, sep=',', header=None)
    data_list = data.to_numpy().tolist()
    attr = [row[1:] for row in data_list]
    cl = [row[0] for row in data_list]
    attributes = np.array([[float(value) for value in row] for row in attr])
    classes = np.array([float(value) for value in cl])
    return attributes, classes


def split_data(attributes, classes, test_size=0.3, random_state=42):
    return train_test_split(attributes, classes, test_size=test_size, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)
    print("Confusion Matrix:\n", conf_matrix)
    return accuracy, class_report, conf_matrix, y_pred


# Główna funkcja
def main():
    file_path = 'data/wine.data'
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    rf = train_random_forest(X_train, y_train)

    accuracy, class_report, conf_matrix, y_pred = evaluate_model(rf, X_test, y_test)

    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    plot_feature_importance(rf, feature_names)

    class_names = np.unique(y).astype(str)
    plot_decision_tree(rf, feature_names, class_names)

    plot_confusion_matrix_custom(conf_matrix, class_names)

    # plot_roc_curve(rf, X_test, y_test)


if __name__ == "__main__":
    main()
