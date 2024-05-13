from data_preprocessing import read_data
import pandas as pd
from graph_utils import make_graph
from graph_knn import GraphKNN
from evaluation import evaluate

# Define paths to your data files
business_data_path = r"business_d.csv"
food_data_path = r"food_d.csv"
health_data_path = r"health_d.csv"

# Read data from CSV files
business_data = read_data("business_d.csv")
food_data = read_data("food_d.csv")
health_data = read_data("health_d.csv")

# Concatenate all data frames
total_data = pd.concat([business_data, food_data, health_data], ignore_index=True)

# Prepare training data
train_texts = total_data["text"].tolist()
train_labels = total_data["label"].tolist()
train_graphs = [make_graph(text) for text in train_texts]

# Define the k value for KNN
k = 3

# Train the Graph KNN model
graph_classifier = GraphKNN(k)
graph_classifier.fit(train_graphs, train_labels)

# Prepare test data
test_text = [
    "By following the tips and tricks for beginners you can plan",
    "By following the tips and tricks for beginners you can plan",
    "By following the tips and tricks for beginners you can plan",
    
    "Citigroup posted lower profit as it spent more on severance payments",
        "Citigroup posted lower profit as it spent more on severance payments",
            "Citigroup posted lower profit as it spent more on severance payments"
    
    "research article that examined 30 years",
    "research article that examined 30 years",
    "research article that examined 30 years",
    
]
test_graphs = [make_graph(text) for text in test_text]

# Predict labels for test data
predictions = [graph_classifier.predict(graph) for graph in test_graphs]

# Get unique class labels from training data
classes = list(set(total_data["label"]))

# Evaluate the model's performance
evaluate(test_labels=classes, predictions=predictions, classes=classes)
