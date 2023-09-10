import torch
from flask import Flask
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex
import glob
import json
import os
import pymongo

model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for your words or phrases
# Open the text file for reading

# List of file names or paths

# Iterate over the list of file names
for filename in glob.glob(f"{textfiles}/*.txt"):
    print(filename)
    with open(filename, "r") as file:
        content = file.read()
        # Insert content into MongoDB collection
        db["documents"].insert_one({"filename": filename, "content": content})



for file_name in file_names:
    try:
        # Open the file in read mode (you can use 'w' for write mode or 'a' for append mode)
        with open(file_name, 'r') as file:
            # Read and process the file content here
            content = file.read()
            files_compiled.append(content);
    except FileNotFoundError:
        print(f"{file_name} not found.")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")
        file.close();

embeddings = model.encode(files_compiled, convert_to_tensor=True) # words --> array from reading text file

# Initialize Annoy index
num_dimensions = len(embeddings[0])  # Assuming embeddings is a list of SBERT embeddings
annoy_index = AnnoyIndex(num_dimensions, metric='euclidean')

# Add SBERT embeddings to the Annoy index with unique identifiers
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build the index
annoy_index.build(n_trees=10)  # Adjust n_trees as needed

# Query the Annoy index to find nearest neighbors
user_input = input("Enter something: ")
print("You entered:", user_input)
query_embedding = model.encode(user_input, convert_to_tensor=True)  # Encode your query
num_neighbors = 10  # Adjust the number of neighbors as needed

# Find nearest neighbors
num_neighbors = 5
neighbor_indices, neighbor_distances = annoy_index.get_nns_by_vector(query_embedding, num_neighbors, include_distances=True)

# Use a set to store unique semantic embeddings
unique_embeddings = set()

# Retrieve unique semantic embeddings
for index in neighbor_indices:
    unique_embeddings.add(tuple(embeddings[index]))

# Convert the set back to a list if needed
unique_embeddings_list = list(unique_embeddings)

# Get the words corresponding to the nearest neighbor indices; Make sure unique_embeddings_list is being use 
nearest_file = [files_compiled[i] for i in neighbor_indices]

# Print the nearest neighbors and their distances
for file, distance in zip(nearest_file, neighbor_distances):
    file_path = "VS Code Flask"
    print(f'File: {file.name}, Similarity Score: {1 - distance}')