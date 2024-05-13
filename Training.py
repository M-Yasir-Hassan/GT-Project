import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, confusion_matrix

# Read data from CSV files
business_df = pd.read_csv("business_d.csv", delimiter=",")
food_df = pd.read_csv("food_d.csv", delimiter=",")
health_df = pd.read_csv("health_d.csv", delimiter=",")

# Concatenate all dataframes
combine_data = pd.concat([business_df,health_df ,food_df], ignore_index=True)
# print(combine_data)
# Preprocessing function
def pre_process(text):
    if isinstance(text, str):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return " ".join(tokens)
    else:
        return "doctor"

# Preprocess text data
combine_data['text'] = combine_data['text'].apply(pre_process)

# Make a Directed Graph according to the paper
def make_graph(string):
    chunks = string.split()
    G = nx.DiGraph()
    for chunk in set(chunks):
        G.add_node(chunk)
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G

def graph_distance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())

class GraphKNN:
    def __init__(self, k:int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []
    
    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels
    
    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = graph_distance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        return prediction

# Plot confusion matrix
def confu_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Prepare training data
train_texts = combine_data['text'].tolist()
train_labels = combine_data['label'].tolist()
train_graphs = [make_graph(text) for text in train_texts]

# Train the model
graph_classifier = GraphKNN(k=3)
graph_classifier.fit(train_graphs, train_labels)

# Test data
test_texts = []
for i in range(0,3):
    test_texts.append(business_df['text'].to_list()[i])
    # print(business_df['text'].to_list()[i])
    test_texts.append(food_df['text'].to_list()[i])
    # print(food_df['text'].to_list()[i])
    test_texts.append(health_df['text'].to_list()[i])
    # print(health_df['text'].to_list()[i])
    print()

# print(test_texts)

test_graphs = [make_graph(text) for text in test_texts]

# Predict
predictions = [graph_classifier.predict(graph) for graph in test_graphs]

# Evaluate
test_labels = ["Business and Finance", "Food","Health and Fitness"]*3
label_order = ["Business and Finance", "Food","Health and Fitness"]

accuracy = accuracy_score(test_labels, predictions)
accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

f1_scores = f1_score(test_labels, predictions, average=None)

f1_score_percentage = f1_scores[0] * 100
print("F1 Score:", "{:.2f}%".format(f1_score_percentage))

jaccard = jaccard_score(test_labels, predictions, average=None)
jaccard_percentage = jaccard[0] * 100
print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))


# Plot confusion matrix
conf_matrix = confusion_matrix(list(test_labels), list(predictions), labels=list(set(test_labels)))

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_order, yticklabels=label_order)

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

