import io

import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from chess import Board
from torch_geometric.nn import GlobalAttention, NNConv

MATE_SCORE = 10000
PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}


def _is_attack(board, from_square, to_square):
    from_piece = board.piece_at(from_square)
    if from_piece is None:
        return False
    to_piece = board.piece_at(to_square)
    return from_square in board.attackers(from_piece.color, to_square) and (
        to_piece is None or to_piece.color != from_piece.color
    )


def _is_defend(board, from_square, to_square):
    from_piece = board.piece_at(from_square)
    to_piece = board.piece_at(to_square)
    if from_piece is None or to_piece is None:
        return False

    return from_square in board.attackers(from_piece.color, to_square) and (
        to_piece.color == from_piece.color
    )


def get_node_features(board):
    """Create node features for each square"""
    features = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Basic square features
        square_features = [
            rank / 7.0,  # Normalized rank (0-1)
            file / 7.0,  # Normalized file (0-1)
            (rank + file) / 14.0,  # Distance from a1
            (rank + (7 - file)) / 14.0,  # Distance from a8
            1.0 if (rank + file) % 2 == 0 else 0.0,  # Square color
            1.0 if board.turn else 0.0,  # Whose turn it is (1 for white, 0 for black)
        ]

        # Piece features (13 dimensions: 6 piece types * 2 colors + piece value)
        piece_features = [0] * 13
        if piece is not None:
            piece_idx = (piece.piece_type - 1) * 2 + (0 if piece.color else 1)
            piece_features[piece_idx] = 1
            # Add piece value
            piece_features[-1] = PIECE_VALUES[piece.symbol().upper()] * (
                1 if piece.color else -1
            )

        features.append(square_features + piece_features)

    return torch.FloatTensor(features)


def get_edge_index_and_features(board):
    """Create edges between squares based on chess relationships"""
    edges = []
    edge_features = []

    for from_square in chess.SQUARES:
        from_piece = board.piece_at(from_square)
        if from_piece is None:
            continue

        # Get all possible moves for the piece
        for to_square in chess.SQUARES:
            # Skip self-connections
            if from_square == to_square:
                continue

            to_piece = board.piece_at(to_square)

            # Check if move is legal
            move = chess.Move(from_square, to_square)
            is_legal = move in board.legal_moves

            # Attack relationship
            is_attack = _is_attack(board, from_square, to_square)

            # Defense relationship (same color pieces)
            is_defend = _is_defend(board, from_square, to_square)

            # Add edges for relevant relationships
            if is_legal or is_attack or is_defend:
                edges.append([from_square, to_square])

                # Edge features
                edge_feat = [
                    float(is_legal),
                    float(is_attack),
                    float(is_defend),
                    1.0 if (to_piece is not None and is_attack) else 0.0,  # Capture
                    1.0 if board.is_check() else 0.0,  # Check
                ]
                edge_features.append(edge_feat)

    edge_index = torch.LongTensor(edges).t().contiguous()
    edge_attr = torch.FloatTensor(edge_features)

    return edge_index, edge_attr


class Model(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim):
        super().__init__()

        # Node embedding layers
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        # Edge networks for each convolution layer
        self.edge_nn1 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim // 2), nn.ReLU()
        )
        self.edge_nn2 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim // 4), nn.ReLU()
        )
        self.edge_nn3 = nn.Sequential(
            nn.Linear(edge_features, hidden_dim * hidden_dim // 4), nn.ReLU()
        )

        # Graph convolution layers
        self.conv1 = NNConv(hidden_dim, hidden_dim // 2, self.edge_nn1)
        self.conv2 = NNConv(hidden_dim // 2, hidden_dim // 2, self.edge_nn2)
        self.conv3 = NNConv(hidden_dim // 2, hidden_dim // 2, self.edge_nn3)

        self.attention = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        self.eval_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        # Encode node features
        x = self.node_encoder(x)

        # Graph convolution layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))

        # Combine features from all layers
        x = x1 + x2 + x3

        # Global attention pooling
        x = self.attention(x, batch)

        # Final evaluation
        return self.eval_net(x).flatten()

    def score(self, pgn, move):
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        for past_move in game.mainline_moves():
            board.push(past_move)
        board.push_san(move)
        node_features = get_node_features(board)
        edge_index, edge_features = get_edge_index_and_features(board)
        evaluation = self.forward(node_features, edge_index, edge_features)
        return evaluation.item()
