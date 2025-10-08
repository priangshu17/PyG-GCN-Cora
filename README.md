# PyG-GCN-Cora

# GCN for Node Classification with PyTorch Geometric

This repository contains a Jupyter Notebook that implements a Graph Convolutional Network (GCN) for semi-supervised node classification. The project uses the **PyTorch Geometric (PyG)** library to build and train the GCN on the classic **Cora** citation network dataset.

## 📝 Overview

Graph Convolutional Networks are a powerful type of neural network designed to work directly with graph-structured data. They learn representations (embeddings) for nodes by aggregating information from their local neighborhoods. This project provides a clear and concise example of how to implement a GCN for a common graph machine learning task: predicting the category of a paper in a citation network.

The notebook covers the full pipeline:
1.  **Data Loading**: Using PyG's built-in `Planetoid` dataset to load the Cora graph.
2.  **Model Definition**: Building a simple two-layer GCN using PyG's `GCNConv` layers.
3.  **Training**: Training the model in a semi-supervised fashion on the training nodes.
4.  **Evaluation**: Testing the model's accuracy on the unseen test nodes.

## 📊 Dataset: The Cora Network

The Cora dataset is a standard benchmark for graph-based machine learning. It is a citation network where:
* **Nodes**: Represent scientific papers.
* **Edges**: Represent citations between papers.
* **Node Features**: A bag-of-words vector for each paper.
* **Task**: To classify each paper into one of seven distinct subject categories.
