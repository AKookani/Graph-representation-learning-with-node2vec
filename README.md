# Graph-representation-learning-with-node2vec

This repository contains two projects focusing on **graph representation learning**. Each project demonstrates different approaches to learning low-dimensional vector representations of nodes in a graph, which can then be used for various machine learning tasks, such as node classification or link prediction.

## Project 1: Movie Recommendation using Graph Embeddings

This project focuses on learning embeddings from a movie-user interaction graph derived from the MovieLens dataset. It applies the **Skip-Gram** approach, using **random walks** on the graph to generate node sequences, which are then used to train a neural network model to learn node (movie) embeddings.

### Key Steps:
- **Data Preprocessing**: Downloads the MovieLens dataset and filters high-rated movies.
- **Graph Construction**: Constructs a weighted movie graph based on pairwise movie ratings by users.
- **Random Walk Generation**: Generates random walks to sample node sequences from the graph, using parameters `p` and `q` to control the likelihood of returning to the previous node or moving forward.
- **Skip-Gram Model Training**: Uses TensorFlow and Keras to implement a Skip-Gram model for learning movie embeddings from the generated random walks.
- **Visualization**: Exports the learned embeddings for visualization in external tools.

### Dataset:
- **MovieLens Small** dataset is used, which contains around 100k ratings for movies.

### Output:
- **Movie Embeddings**: The model learns movie embeddings which are saved as `embeddings.tsv`.
- **Metadata**: Movie metadata (titles) are saved in `metadata.tsv` for visualization in embedding projectors.

### Technologies Used:
- Python, NetworkX, TensorFlow, Keras, NumPy, pandas, TQDM

---

## Project 2: Node2Vec on Cora Dataset

The second project demonstrates the **DeepWalk** and **Node2Vec** algorithms applied to the **Cora citation network dataset**. These methods generate random walks on the graph and apply the **Word2Vec** model to learn embeddings for each node.

### Key Steps:
- **Random Walks**: Uses a custom `DeepWalk` class to generate random walks on the graph. Both uniform and weighted random walks are supported.
- **Node2Vec**: Utilizes **PyTorch Geometric's Node2Vec** to efficiently compute node embeddings on the Cora dataset.
- **Training**: The Node2Vec model is trained using **PyTorch Geometric**, with an emphasis on negative sampling for computational efficiency.

### Dataset:
- **Cora** citation dataset, a common benchmark dataset in graph learning.

### Output:
- **Node Embeddings**: The learned node embeddings are saved as `vectors.txt`.
- **Node Labels**: Node labels from the Cora dataset are saved as `labels.txt`.

### Technologies Used:
- Python, NetworkX, Gensim, PyTorch, PyTorch Geometric

## Dependencies

- TensorFlow
- Keras
- PyTorch
- PyTorch Geometric
- NetworkX
- Gensim
- NumPy
- pandas
