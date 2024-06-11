import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import graphviz
from sklearn.tree import export_graphviz
import seaborn as sns


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


def plot_feature_importance(rf, feature_names, file_name='raport/feature_importance.png'):
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_decision_tree(rf, feature_names, class_names, file_name='raport/decision_tree'):
    tree = rf.estimators_[0]
    dot_data = export_graphviz(tree, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(cls) for cls in class_names],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(file_name)  # Save as .pdf


def plot_confusion_matrix_custom(conf_matrix, class_names, file_name='raport/confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_roc_curve(rf, X_test, y_test, file_name='raport/roc_curve.png'):
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


# Główna funkcja
def main():
    file_path = 'wine.data'
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
