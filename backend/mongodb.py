import torch
from flask import Flask
from sentence_transformers import SentenceTransformer, util
from annoy import AnnoyIndex
import glob
import json
import os
import pymongo
import numpy as np

# Connect Python file to MongoDB
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = "mongodb+srv://sylviajacoby:Ellaella$2@cluster0.3cfxdom.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    
# Connect Python to a specific database
from pymongo.database import Database
db = client["HackDuke"]

# Define the directory containing your text files
text_files_directory = '/users/kristinaschaufele/Desktop/hackduke/textfiles/'
print(text_files_directory)

# Delete all existing documents in the collection
db["documents"].delete_many({})

# Iterate over the files in the directory
for filename in glob.glob(f"{text_files_directory}/*.txt"):
    print(filename)
    try:
        with open(filename, "r") as file:
            content = file.read()
        # Insert content into MongoDB collection
        db["documents"].insert_one({"filename": filename, "content": content});
    except FileNotFoundError:
        print(f"{filename} not found.")
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")
        file.close();
        
model = SentenceTransformer('all-MiniLM-L6-v2')

alldocs = db["documents"].find()
files_compiled = np.array(alldocs);

embeddings = model.encode(files_compiled, convert_to_tensor=True) # words --> array from reading text file

# Initialize Annoy index
num_dimensions = len(embeddings[0])  # Assuming embeddings is a list of SBERT embeddings
annoy_index = AnnoyIndex(num_dimensions, metric='euclidean')

# Add SBERT embeddings to the Annoy index with unique identifiers
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)
    

# Build the index
annoy_index.build(n_trees=10)  # Adjust n_trees as needed

data_documents = alldocs.find({"file_data": {"$exists": True}})

for document in data_documents:
    # Assuming you have a document like this
    document = {
        "_id": 1,
        "text": "This is the text content of a document.",
        "_embedding": embedding.tolist()  # SBERT embedding
    }

    # Add the Annoy index item ID to the document
    document["annoy_id"] = annoy_index.get_nns_by_vector(document["_embedding"], 1)[0]  # Assuming 1 nearest neighbor

    # Insert or update the document in MongoDB
    alldocs.update_one({"_id": 1}, {"$set": document}, upsert=True)

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
nearest_file = [unique_embeddings_list[i] for i in neighbor_indices]

# Print the nearest neighbors and their distances
for file, distance in zip(nearest_file, neighbor_distances):
    file_path = "hackduke"
    print(f'File: {file.name}, Similarity Score: {1 - distance}')