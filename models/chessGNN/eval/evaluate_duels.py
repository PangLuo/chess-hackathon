"""
Chess Game Simulation and Evaluation Script

This script facilitates the evaluation of chess-playing agents by simulating games between them.
It loads trained models, executes multiple chess games, and records evaluation metrics.

Usage:
In this folder, create two subfolders, each containing model.py, model_config.yaml
and checkpoint.pt. Then, update module1_name and module2_name with the names of
these subfolders.

Run the script directly—the results will be saved as results.csv, which will be used
by report.py.
"""

import importlib
import multiprocessing as mp
import os
import time

import pandas as pd
import torch
import yaml
from chess import Board

import chess_gameplay as chg
from chess_gameplay import evaluate_position, sans_to_pgn


def get_selected_move_san(board, moves_san, agent):
    legal_moves = list(board.legal_moves)
    # This function is supposed to be called in fast_evaluate,
    # which already checks whether there is a checkmate or
    # stalemate. So legal_moves can't be empty here.
    assert legal_moves != []
    legal_moves_san = [board.san(move) for move in legal_moves]

    pgn = sans_to_pgn(moves_san)
    selected_move_san = agent.select_move(pgn, legal_moves_san)
    return selected_move_san


def get_model(module_name, module):
    model_config = yaml.safe_load(open(os.path.join(module_name, "model_config.yaml")))
    model = module.Model(**model_config)
    checkpoint = torch.load(
        os.path.join(module_name, "checkpoint.pt"), map_location="cpu"
    )
    model.load_state_dict(checkpoint["model"])
    return model


def fast_evaluate(agents, max_rounds=50, eval_time_limit=0.1, eval_depth_limit=10):
    # board will align with moves_san
    board = Board()
    moves_san = []
    scores = {"white": [], "black": [], "round": [], "ply": []}
    rounds = 0

    while True:
        player = "white" if board.turn else "black"
        opponent = "black" if player == "white" else "white"
        selected_move_san = get_selected_move_san(board, moves_san, agents[player])

        moves_san.append(selected_move_san)
        board.push_san(selected_move_san)

        score = evaluate_position(
            board,
            time_limit=eval_time_limit,
            depth_limit=eval_depth_limit,
            STOCKFISH_PATH="/root/chess-hackathon/utils/stockfish",
        )

        scores[player].append(-score)
        scores[opponent].append(score)
        scores["round"].append(rounds)
        scores["ply"].append(selected_move_san)

        # Check if is_checkmate or is_stalemate is true after
        # each move.
        if board.is_checkmate() or board.is_stalemate():
            break

        if player == "black":
            rounds += 1
            if rounds >= max_rounds:
                break

    return scores


def process_game(game, agents, max_rounds, eval_time_limit, eval_depth_limit):
    print(f"processing game {game}")
    start = time.perf_counter()
    scores = pd.DataFrame(
        fast_evaluate(
            agents,
            max_rounds=max_rounds,
            eval_time_limit=eval_time_limit,
            eval_depth_limit=eval_depth_limit,
        )
    )
    scores["game"] = game
    end = time.perf_counter()
    print(f"game {game} processed. Time spent: {end - start} seconds\n")
    return scores


if __name__ == "__main__":
    games = 2

    module1_name = "G"
    module2_name = "C1"

    module1 = importlib.import_module(f"{module1_name}.model")
    module2 = importlib.import_module(f"{module2_name}.model")

    model1 = get_model(module1_name, module1)
    model2 = get_model(module2_name, module2)

    agents = {"white": chg.Agent(model1), "black": chg.Agent(model2)}

    start = time.perf_counter()

    # Not every model works with multiprocessing.
    try:
        num_workers = mp.cpu_count()
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                process_game,
                [
                    (game, agents, 50, 0.1, 10)
                    for game in range(games)  # 50 rounds, 0.1s, 10 depth
                ],
            )
    except:
        results = []
        for game in range(games):
            results.append(process_game(game, agents, 50, 0.1, 10))

    end = time.perf_counter()
    print(
        f"Total time spent: {end - start} seconds",
    )
    print("\nsaving results\n")
    concat_results = pd.concat(results, ignore_index=True)
    concat_results.to_csv("results.csv", index=False)

    for player in ("white", "black"):
        print(f"{player} mean: {concat_results[player].mean()}")
        print(f"{player} std: {concat_results[player].std()}")
