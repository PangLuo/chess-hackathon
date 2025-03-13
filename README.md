# chess-gnn

## Acknowledgement
This is a fork that implements a GNN for playing chess. The original
repository can be found here:
https://github.com/StrongResearch/chess-hackathon.

The GNN model was trained using fully managed GPUs on the Strong
Compute Instant Super Computer system. I appreciate the sponsorship
and support from Adam and Tim at Strong Compute.

### Evaluation

This table presents the performance of the GNN model playing chess against various opponents as both White and Black. Each matchup consisted of 100 games, with a maximum of 50 moves (100 plies) per game. If a game did not end in checkmate or stalemate within 50 moves, Stockfish was used to evaluate the final board position, and the side with a positive score was considered the winner.

| Matchup              | W (Checkmates)     | D  | L (Checkmates)     | G Centipawn Mean | G Centipawn Std Dev |
|----------------------|-------------------|----|-------------------|------------------|--------------------|
| G vs C1 (G as White) | 71 (checkmate: 10) | 2  | 27 (checkmate: 3)  | -9               | 1592               |
| G vs C1 (G as Black) | 56 (checkmate: 11) | 2  | 42 (checkmate: 2)  | -132             | 1731               |
| G vs C2 (G as White) | 58 (checkmate: 11) | 2  | 40 (checkmate: 6)  | 34               | 1399               |
| G vs C2 (G as Black) | 65 (checkmate: 20) | 2  | 33 (checkmate: 2)  | 64               | 1426               |
| G vs C3 (G as White) | 99 (checkmate: 58) | 1  | 0                  | 1349             | 2494               |
| G vs C3 (G as Black) | 100 (checkmate: 65)| 0  | 0                  | 1413             | 2635               |
| G vs T (G as White)  | -                 | -  | -                  | -                | -                  |
| G vs T (G as Black)  | -                 | -  | -                  | -                | -                  |

- **G**: the GNN model (rank correlation: 75%). Rank correlation refers to Spearman's rank correlation coefficient computed between the target values and the predicted values in a batch. It serves as a measure of how closely the model's predictions align with Stockfish's evaluation.
- **C1**: the CNN model that won the November 2024 hackathon
- **C2**: a model trained using the checkpoint of C1 (rank correlation: 86%)
- **C3**: a CNN model trained from scratch during the January 2025 hackathon (rank correlation: 33%)
- **T**: a transformer model (rank correlation: 92%)
- **W**: Number of wins
- **D**: Number of draws
- **L**: Number of losses
- **Checkmate**: Number of wins/losses by checkmate
- **G Centipawn Mean**: Average centipawn evaluation for G
- **G Centipawn Std Dev**: Standard deviation of centipawn evaluations

### Model Architecture Summary
This model is a Graph Neural Network (GNN) based on PyTorch Geometric designed to evaluate chess board positions by predicting a scalar evaluation score. The architecture processes a chess board as a graph, where each square is a node, and edges represent piece interactions such as legal moves, attacks, and defenses.

#### 1. Input Representation
**Node Features**
Each of the 64 board squares has a feature vector encoding:
- Board position (rank, file, square color)
- Occupying piece type and value (if any)
- Turn information (who is to move)

**Edge Features**
- Legal moves, attacks, and defenses
- Whether a capture or check occurs

#### 2. Model Architecture
**Node Encoding**
A fully connected MLP processes node features.

**Edge Encoding**
Three separate MLPs encode edge features for three layers of NNConv (a message-passing convolution layer).

**Graph Processing**
3 NNConv layers process the node and edge features using the encoded edge attributes. The output of these layers is combined via residual connections.

**Global Pooling**
A Global Attention layer aggregates node features into a graph-level representation.

**Final Evaluation**
A feedforward network predicts a scalar evaluation score.

#### 3. Output
The model outputs a single scalar value representing the evaluation of the chess position. The .score(pgn, move) function allows direct evaluation of a move in a given chess game.

### Parameter Count
This model has about 2 million parameters, which is about half of C1/C2/C3.

### Training Dataset
The dataset used for training consists of 5.7 million positions, which is much smaller than the one used for training C1/C2/C3 (35.4 million positions). The details of how this dataset was created are as follows.

**Data Source**
Uses Lichess Elite (games by players rated 2400+ against players rated 2200+) PGN files from June 2020 to August 2022.

**Graph Representation**
Converts chess positions into graph-based formats with PyTorch Geometric.

**Processing Pipeline**
- Downloads PGN files.
- Gets scores for chess boards using Stockfish and uses them as target values.
- Converts board states into graph structures.
- Stores them as HDF5 files.

**Sampling & Scaling**
- Limits max positions per PGN.
- Allows score scaling (log or linear) to facilitate training.
- Uses a sample rate to select positions from games.

### Training Method
The training method is the same as the one used in the original repository, except that I started training the model with a small dataset of 300K positions. Whenever I observed that the validation loss stopped decreasing and the validation rank correlation stopped increasing, I doubled the sizes of the training and validation datasets.

To prevent previous training data from leaking into the new validation data, the original training and validation sets were kept intact. New data were sampled from games played over the following months and added to the training and validation sets. Interestingly, the validation loss and validation rank correlation did not change significantly when the validation dataset was doubled, suggesting that the model generalizes reasonably well to unseen data.
