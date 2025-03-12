# chess-gnn

## Acknowledgement
This is a fork that implements a GNN for playing chess. The original
repository can be found here:
https://github.com/StrongResearch/chess-hackathon.

### Step X. Evaluation

This table presents the performance of the GNN model playing chess against various opponents as both White and Black.

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

- **G**: the GNN model (rank correlation: 75%)
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
