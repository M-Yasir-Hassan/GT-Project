import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix

# Read data from CSV files
business_d = pd.read_csv("business_d.csv",  encoding="latin1")
food_d = pd.read_csv("food_d.csv",  encoding="latin1")
health_d = pd.read_csv("health_d.csv",  encoding="latin1")


# Concatenate all data frames
totaldata = pd.concat([business_d, food_d, health_d], ignore_index=True)


# preprocess_funing function
def preprocess_fun(text):
    if isinstance(text, str):
        # Tokenization
        tokens = re.findall(r"\b\w+\b", text.lower())
        # Stop-word removal and stemming can be added here if needed
        return " ".join(tokens)
    else:
        return "doctor"


# Make a Directed Graph according to the paper
def makeGraph(string):
    # Split the string into words
    chunks = string.split()
    # Create a directed graph
    G = nx.DiGraph()
    # Add nodes for each unique word
    for chunk in set(chunks):
        G.add_node(chunk)
    # Add edges between adjacent words
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G


# Calculate graph distance
def graphDistance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())


class GraphKNN:
    def __init__(self, k: int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []

    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels

    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = graphDistance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[
            : self.k
        ]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        # print("Prediction:", prediction)
        return prediction


# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# preprocess_fun text data
totaldata["text"] = totaldata["text"].apply(preprocess_fun)

# Prepare training data
trainTexts = totaldata["text"].tolist()
trainLabels = totaldata["label"].tolist()
trainGraphs = [makeGraph(text) for text in trainTexts]

# Train the model
graphClassifier = GraphKNN(k=3)
graphClassifier.fit(trainGraphs, trainLabels)

# Test data
testText = [
    "By following the tips and tricks for beginners you can plan",
    "Citigroup posted lower profit as it spent more on severance payments",
    "research article that examined 30 years",
]
testGraphs = [makeGraph(text) for text in testText]

# Predict
predictions = [graphClassifier.predict(graph) for graph in testGraphs]

# Evaluate
testLabels = ["Food", "Business and Finance", "Health and Fitness"]
# Calculate accuracy
accuracy = accuracy_score(testLabels, predictions)
# Calculate accuracy
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# Calculate F1 Score for the second class
f1Scores = f1_score(testLabels, predictions, average=None)
# print("F1 Scores:", f1_scores)
f1ScorePercentage = f1Scores[0] * 100
print("F1 Score:", "{:.2f}%".format(f1ScorePercentage))

# Calculate Jaccard similarity for the second class
jaccard = jaccard_score(testLabels, predictions, average=None)
# print("jaccard:",jaccard)
jaccard_percentage = jaccard[0] * 100
print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))


# Plot confusion matrix
confMatrix = confusion_matrix(
    list(testLabels), list(predictions), labels=list(set(testLabels))
)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(
    confMatrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=set(testLabels),
    yticklabels=set(testLabels),
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
