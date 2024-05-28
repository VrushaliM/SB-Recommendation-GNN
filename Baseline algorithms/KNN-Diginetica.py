import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import defaultdict


# Load the dataset
data = pd.read_csv('data1.csv')#diginetica data

# Preprocess the data
def preprocess_data(data):
    data['timestamp'] = pd.to_datetime(data['eventdate'] + ' ' + data['timeframe'].astype(str), errors='coerce')
    data = data.drop(columns=['eventdate', 'timeframe'])
    data = data.dropna(subset=['timestamp'])
    return data

data = preprocess_data(data)

# Display the first few rows
print(data.head())


# Sort the data by session_id and timestamp
data.sort_values(by=['session_id', 'timestamp'], inplace=True)

#group data by session_id
sessions = data.groupby('session_id')['item_id'].apply(list).reset_index()

# generate sequences of items for each session
def generate_sequences(session, seq_length):
    sequences = []
    for i in range(len(session) - seq_length):
        sequences.append(session[i:i+seq_length+1])
    return sequences

#fenerate sequences of items for each session
seq_length = 3 
session_sequences = []
for session in sessions['item_id']:
    session_sequences.extend(generate_sequences(session, seq_length))

#unique items
unique_items = sorted(list(set(data['item_id'])))

#convert sequences to binary matrix
binary_matrix = []
targets = []
for seq in session_sequences:
    binary_vector = [1 if item in seq[:-1] else 0 for item in unique_items]
    binary_matrix.append(binary_vector)
    targets.append(seq[-1])

# training
X = binary_matrix  # Features
y = targets  # Target (next item)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#training KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='jaccard')  # Adjust parameters as needed
knn_model.fit(X_train, y_train)

#evaluate the model
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

#calculate Mean Reciprocal Rank
def calculate_mrr(predictions, targets):
    reciprocal_ranks = []
    for prediction, target in zip(predictions, targets):
        if isinstance(target, int):
            target = [target]  # Convert to list if it's a single integer
        if prediction in target:
            rank = 1 / (target.index(prediction) + 1)
            reciprocal_ranks.append(rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

predictions = knn_model.predict(X_test)
mrr = calculate_mrr(predictions, y_test)
print("Mean Reciprocal Rank (MRR):", mrr)

custom_test_sequence = [81766, 31331, 32118]  # Example custom test sequence from your data

# Convert the custom test sequence to the same format as the training data
custom_test_vector = [1 if item in custom_test_sequence else 0 for item in unique_items]
next_item_prediction = knn_model.predict([custom_test_vector])
print("Predicted next item in the custom test sequence:", next_item_prediction[0])
