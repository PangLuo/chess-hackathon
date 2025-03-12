"""
Chess Data Preparator for Graph-Based Neural Networks

This module provides functionality for processing chess games from PGN files,
evaluating board positions using the Stockfish engine, and converting board positions
into graph data structures suitable for training graph neural networks (GNNs).

Classes:
    ChessDataPreparator: Handles the extraction, evaluation, and graph transformation of chess positions.

Dependencies:
    - chess.pgn: For parsing PGN files.
    - torch: For tensor operations.
    - torch_geometric.data: For graph data representation.
    - Stockfish (via python-chess): For position evaluations.

Main Features:
    - Loads and processes PGN files from a ZIP archive.
    - Evaluates board positions using Stockfish with configurable scaling.
    - Caches evaluations to optimize performance.
    - Converts chess board states into graph representations.

Usage:
    preparator = ChessDataPreparator(stockfish_path="/path/to/stockfish")
    positions = preparator.process_pgn("games.zip", max_positions=1000)
    graph_data = preparator.create_graph_data(positions[0]['board'])
"""

import io
import random
import zipfile
from collections import defaultdict

import chess.pgn
import torch
from torch_geometric.data import Data

from models.chessGNN.model import (
    MATE_SCORE,
    get_edge_index_and_features,
    get_node_features,
)


class ChessDataPreparator:
    def __init__(self, stockfish_path):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Threads": 1})
        self.position_cache = {}

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.quit()

    def get_position_evaluation(self, board, score_scaling="log", depth_limit=10):
        """Get Stockfish evaluation for a position"""
        # Use FEN as cache key (excluding move counters)
        fen = " ".join(board.fen().split(" ")[:-2])

        if fen in self.position_cache:
            return self.position_cache[fen]

        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth_limit),
            )
            score = torch.tensor(
                info["score"].relative.score(mate_score=MATE_SCORE), dtype=torch.float
            )
            # The reason for negating `score` when calculating `scaled_score` is as follows:
            # Assume `board.turn` is `True`, meaning it's White's turn, and the previous
            # board state is `prev_board`.
            # `score` represents the engine's evaluation of the best move based on `board`
            # from White's perspective. This evaluation can also be interpreted as the
            # assessment of `board` from White's perspective.
            # Therefore, the evaluation of `board` from Black's perspective should be `-score`.
            # We actually save the evaluation of `board` from the perspective of the opponent
            # of `board.turn`. When selecting the next move at `prev_board`, the opponent of
            # `board.turn` will evaluate each possible `board` and choose the move that results
            # in a `board` with a high `-score` (i.e., a low `score`).
            if score_scaling == "log":
                scaled_score = -torch.sign(score) * torch.log1p(torch.abs(score))
            elif score_scaling == "linear":
                scaled_score = -score / 100
            else:
                raise ValueError("score_scaling must be either log or linear")

            self.position_cache[fen] = scaled_score
            return scaled_score

        except Exception as e:
            print(f"Error evaluating position: {e}")
            return 0.0

    def process_pgn(
        self, pgn_path, max_positions=1000, score_scaling="log", sample_rate=0.1
    ):
        """Process PGN file and extract positions with engine evaluations"""
        positions = []
        position_counts = defaultdict(int)

        with zipfile.ZipFile(pgn_path, "r") as zip_file:
            for file_name in zip_file.namelist():
                if not file_name.endswith(".pgn"):
                    continue
                with zip_file.open(file_name) as pgn_file:
                    pgn_stream = io.TextIOWrapper(pgn_file, encoding="utf-8")
                    while True:
                        game = chess.pgn.read_game(pgn_stream)
                        if game is None:
                            break

                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                            # Get position FEN (excluding move counters)
                            fen = " ".join(board.fen().split(" ")[:-2])
                            if (
                                random.random() > sample_rate
                                or position_counts[fen] >= 3
                            ):
                                continue
                            position_counts[fen] += 1
                            evaluation = self.get_position_evaluation(
                                board, score_scaling
                            )
                            positions.append(
                                {
                                    "board": board.copy(),
                                    "evaluation": evaluation,
                                    "fen": fen,
                                }
                            )
                            if len(positions) >= max_positions:
                                return positions

        return positions

    def create_graph_data(self, board):
        """Convert a board position into graph data"""
        node_features = get_node_features(board)
        edge_index, edge_attr = get_edge_index_and_features(board)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return data
