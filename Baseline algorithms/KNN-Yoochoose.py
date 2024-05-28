import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load the dataset
data = pd.read_csv('yoochoose1.csv')

#convert timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

#sort the data by session_id and timestamp
data.sort_values(by=['session_id', 'timestamp'], inplace=True)

#group data by session_id
sessions = data.groupby('session_id')['item_id'].apply(list).reset_index()

#function to generate sequences of items for each session
def generate_sequences(session, seq_length):
    sequences = []
    for i in range(len(session) - seq_length):
        sequences.append(session[i:i+seq_length])
    return sequences

#generate sequences of items for each session
seq_length = 3
session_sequences = []
for session in sessions['item_id']:
    session_sequences.extend(generate_sequences(session, seq_length))

#get unique items
unique_items = sorted(list(set(data['item_id'])))

#convert sequences to binary matrix
binary_matrix = []
for seq in session_sequences:
    binary_vector = [1 if item in seq else 0 for item in unique_items]
    binary_matrix.append(binary_vector)

# Prepare data for training
X = binary_matrix[:-1]  # Features
y = [seq[-1] for seq in session_sequences[1:]]  # Target (next item)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='jaccard')
knn_model.fit(X_train, y_train)

# Evaluate the model
y_pred = knn_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Calculate Mean Reciprocal Rank
def calculate_mrr(predictions, targets):
    reciprocal_ranks = []
    for prediction, target in zip(predictions, targets):
        if isinstance(target, int):
            target = [target]
        if prediction in target:
            rank = 1 / (target.index(prediction) + 1)
            reciprocal_ranks.append(rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


test_sessions = [[214718167, 214820261, 214821017], [214718167, 214820261, 214821017, 214530776]]
predictions = knn_model.predict(X_test)
mrr = calculate_mrr(predictions, y_test)
print("Mean Reciprocal Rank (MRR):", mrr)

custom_test_sequence = [214718167, 214820261]  # Example custom test sequence

# Convert the custom test sequence to the same format as the training data
custom_test_vector = [1 if item in custom_test_sequence else 0 for item in unique_items]
next_item_prediction = knn_model.predict([custom_test_vector])
print("Predicted next item in the custom test sequence:", next_item_prediction[0])
