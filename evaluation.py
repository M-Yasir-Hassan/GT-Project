from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def evaluate(test_labels, predictions, classes):
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    accuracy_percentage = accuracy * 100
    print("Accuracy: {:.2f}%".format(accuracy_percentage))

    # Calculate F1 Score for the second class
    f1_scores = f1_score(test_labels, predictions, average=None)
    f1_score_percentage = f1_scores[1] * 100
    print("F1 Score:", "{:.2f}%".format(f1_score_percentage))

    # Calculate Jaccard similarity for the second class
    jaccard = jaccard_score(test_labels, predictions, average=None)
    jaccard_percentage = jaccard[1] * 100
    print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))

    # Plot confusion matrix
    conf_matrix = confusion_matrix(list(test_labels), list(predictions), labels=list(set(test_labels)))
    plot_confusion_matrix(test_labels, predictions, classes)
