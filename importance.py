import numpy as np
from matplotlib import pyplot as plt
from pygments.lexers import graphviz
from sklearn.tree import export_graphviz

from main import rf

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Visualize one of the trees in the forest
tree = rf.estimators_[0]
dot_data = export_graphviz(tree, out_file=None,
                           feature_names=feature_names,
                           class_names=[str(cls) for cls in np.unique(y)],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")


y_prob = rf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


def plot():
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def importance(rf):

    plot()
