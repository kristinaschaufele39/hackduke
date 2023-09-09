import torch
from flask import Flask
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex

# Load a pre-trained SBERT model (you can replace 'bert-base-nli-stsb-mean-tokens' with your preferred model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for your words or phrases
# Open the text file for reading
file_path = 'melville.txt'  # Replace 'your_file.txt' with the path to your text file
with open(file_path, 'r') as file:
    text = file.read()

# Split the text into words
words = text.split()
file.close();

embeddings = model.encode(words, convert_to_tensor=True) # words --> array from reading text file

# Define the number of dimensions in your embeddings
num_dimensions = len(embeddings[0])

# Initialize the Annoy index
annoy_index = AnnoyIndex(num_dimensions, metric='angular')

# Add the embeddings to the Annoy index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build the index
annoy_index.build(n_trees=10)  # Adjust the number of trees as needed for your dataset

# Compute the embedding for the query & Get input from the user
user_input = input("Enter something: ")
# Print the user's input
print("You entered:", user_input)
query = user_input # replace with user input
query_embedding = model.encode(query, convert_to_tensor=True)

# Find the nearest neighbors (adjust 'num_neighbors' as needed)
num_neighbors = 5
neighbor_indices, neighbor_distances = annoy_index.get_nns_by_vector(query_embedding, num_neighbors, include_distances=True)

# Get the words corresponding to the nearest neighbor indices
nearest_words = [words[i] for i in neighbor_indices]

# Print the nearest neighbors and their distances
for word, distance in zip(nearest_words, neighbor_distances):
    print(f'Word: {word}, Similarity Score: {1 - distance}')