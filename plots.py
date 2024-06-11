import graphviz
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc


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
