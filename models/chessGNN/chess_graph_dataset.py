"""
Chess Graph Dataset for PyTorch Geometric.

This module defines `ChessGraphDataset`, a PyTorch Geometric `Dataset` that processes chess games from PGN files
into graph-based representations suitable for machine learning models. The dataset downloads raw PGN files,
extracts chess positions using Stockfish, and stores them as graphs in HDF5 format.

Classes:
    - ChessGraphDataset: A dataset that loads and processes chess positions as graph structures.

Dependencies:
    - h5py
    - numpy
    - torch
    - torch_geometric
    - models.chessGNN.chess_data_preparator.ChessDataPreparator

Example Usage:
    dataset = ChessGraphDataset(root="data/chess")
    data = dataset[0]  # Get the first chess position graph.

Attributes:
    - max_positions_per_pgn (int): Max positions extracted per PGN file.
    - score_scaling (str): Scaling method for evaluation scores ("log" or "linear").
    - sample_rate (float): Probability of sampling a position from a game.
    - inventory (list): List of processed HDF5 files and position counts.
    - total (int): Total number of positions in the dataset.
"""

import os
import random
import time
from itertools import accumulate

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, download_url

from models.chessGNN.chess_data_preparator import ChessDataPreparator


class ChessGraphDataset(Dataset):
    def __init__(
        self,
        root,
        max_positions_per_pgn=1000,
        score_scaling="log",  # "log" or "linear"
        sample_rate=0.1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        # This line must be located before super().__init__.
        # The reason is that super().__init__ will call the method
        # process when necessary, and process refers to
        # self.blah.
        self.max_positions_per_pgn = max_positions_per_pgn
        self.score_scaling = score_scaling
        self.sample_rate = sample_rate

        super().__init__(root, transform, pre_transform, pre_filter)
        with open(os.path.join(self.processed_dir, "inventory.txt"), "r") as f:
            self.inventory = f.readlines()
        sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
        self.sizes = [int(s) for s in sizes]
        self.total = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    @property
    def raw_file_names(self):
        return [
            "lichess_elite_2020-06.zip",
            "lichess_elite_2020-07.zip",
            "lichess_elite_2020-08.zip",
            "lichess_elite_2020-09.zip",
            "lichess_elite_2020-10.zip",
            "lichess_elite_2020-11.zip",
            "lichess_elite_2020-12.zip",
            "lichess_elite_2021-01.zip",
            "lichess_elite_2021-02.zip",
            "lichess_elite_2021-03.zip",
            "lichess_elite_2021-04.zip",
            "lichess_elite_2021-05.zip",
            "lichess_elite_2021-06.zip",
            "lichess_elite_2021-07.zip",
            "lichess_elite_2021-08.zip",
            "lichess_elite_2021-09.zip",
            "lichess_elite_2021-10.zip",
            "lichess_elite_2021-11.zip",
            "lichess_elite_2021-12.zip",
            "lichess_elite_2022-01.zip",
            "lichess_elite_2022-02.zip",
            "lichess_elite_2022-03.zip",
            "lichess_elite_2022-04.zip",
            "lichess_elite_2022-05.zip",
            "lichess_elite_2022-06.zip",
            "lichess_elite_2022-07.zip",
            "lichess_elite_2022-08.zip",
        ]

    @property
    def processed_file_names(self):
        return [
            "lichess_elite_2020-06.h5",
            "lichess_elite_2020-07.h5",
            "lichess_elite_2020-08.h5",
            "lichess_elite_2020-09.h5",
            "lichess_elite_2020-10.h5",
            "lichess_elite_2020-11.h5",
            "lichess_elite_2020-12.h5",
            "lichess_elite_2021-01.h5",
            "lichess_elite_2021-02.h5",
            "lichess_elite_2021-03.h5",
            "lichess_elite_2021-04.h5",
            "lichess_elite_2021-05.h5",
            "lichess_elite_2021-06.h5",
            "lichess_elite_2021-07.h5",
            "lichess_elite_2021-08.h5",
            "lichess_elite_2021-09.h5",
            "lichess_elite_2021-10.h5",
            "lichess_elite_2021-11.h5",
            "lichess_elite_2021-12.h5",
            "lichess_elite_2022-01.h5",
            "lichess_elite_2022-02.h5",
            "lichess_elite_2022-03.h5",
            "lichess_elite_2022-04.h5",
            "lichess_elite_2022-05.h5",
            "lichess_elite_2022-06.h5",
            "lichess_elite_2022-07.h5",
            "lichess_elite_2022-08.h5",
        ]

    def download(self):
        for url in (
            "https://database.nikonoel.fr/lichess_elite_2020-06.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-07.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-08.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-09.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-10.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-11.zip",
            "https://database.nikonoel.fr/lichess_elite_2020-12.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-01.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-02.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-03.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-04.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-05.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-06.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-07.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-08.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-09.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-10.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-11.zip",
            "https://database.nikonoel.fr/lichess_elite_2021-12.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-01.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-02.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-03.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-04.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-05.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-06.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-07.zip",
            "https://database.nikonoel.fr/lichess_elite_2022-08.zip",
        ):
            path = download_url(url, self.raw_dir)

    def process(self):
        preparator = ChessDataPreparator("/root/chess-hackathon/utils/stockfish")
        inventory = []
        self.total = 0
        for raw_path in self.raw_paths:
            positions = preparator.process_pgn(
                raw_path,
                max_positions=self.max_positions_per_pgn,
                score_scaling=self.score_scaling,
                sample_rate=self.sample_rate,
            )
            h5_basename = os.path.basename(raw_path).replace(".zip", ".h5")
            count_positions = len(positions)
            inventory.append(f"{count_positions} {h5_basename}\n")
            self.total += count_positions
            with h5py.File(os.path.join(self.processed_dir, h5_basename), "w") as hf:
                for i, pos in enumerate(positions):
                    if self.pre_filter is not None and not self.pre_filter(pos):
                        continue
                    if self.pre_transform is not None:
                        pos = self.pre_transform(pos)
                    data = preparator.create_graph_data(pos["board"])
                    grp = hf.create_group(f"data_{i}")
                    grp.create_dataset("edge_index", data=data.edge_index.numpy())
                    grp.create_dataset("edge_attr", data=data.edge_attr.numpy())
                    grp.create_dataset("x", data=data.x.numpy())
                    grp.create_dataset("y", data=pos["evaluation"].view(-1).numpy())
                    string_dtype = h5py.string_dtype(encoding="utf-8")
                    grp.create_dataset(
                        "fen", data=np.array(pos["fen"], dtype=string_dtype)
                    )
            print(f"Dataset saved to {h5_basename}")
        inventory.insert(0, f"Total pgns: {self.total}\n")
        with open(os.path.join(self.processed_dir, "inventory.txt"), "w") as f:
            f.writelines(inventory)
        print("inventory created")

    def len(self):
        return self.total

    def get(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        board_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.processed_dir, self.filenames[hdf_idx])
        with h5py.File(hdf_path, "r") as hf:
            grp = hf[f"data_{board_idx}"]
            edge_index = torch.tensor(grp["edge_index"][:], dtype=torch.long)
            edge_attr = torch.tensor(grp["edge_attr"][:], dtype=torch.float)
            x = torch.tensor(grp["x"][:], dtype=torch.float)
            y = torch.tensor(grp["y"], dtype=torch.float).view(-1)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


if __name__ == "__main__":
    start = time.time()
    random.seed(42)
    dataset = ChessGraphDataset(
        root="chess_dataset_hdf5",
        max_positions_per_pgn=1100000,
        score_scaling="log",
    )
    print(f"Dataset with {len(dataset)} graphs/positions created/loaded")
    print(f"time spent: {round(time.time() - start)} seconds")
    print("the first graph/position:", dataset[0])
