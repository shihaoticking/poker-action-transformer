import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import process_postflop_dataset, process_preflop_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--preflop_input_path", type=str, required=True)
    parser.add_argument("--postflop_input_path", type=str, required=True)
    parser.add_argument("--output_train_path", type=str, required=True)
    parser.add_argument("--output_test_path", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # preflop
    df = pd.read_csv(
        args.preflop_input_path,
        dtype=str,
        usecols=["prev_line", "hero_holding", "correct_decision"],
    )
    df = df.fillna("")
    processed_preflop_df = process_preflop_dataset(df)

    # postflop
    df = pd.read_csv(
        args.postflop_input_path,
        dtype=str,
        usecols=[
            "preflop_action",
            "postflop_action",
            "correct_decision",
            "evaluation_at",
            "holding",
            "board_flop",
            "board_turn",
            "board_river",
        ],
    )
    df = df.fillna("")
    processed_postflop_df = process_postflop_dataset(df)

    # split train and test
    # 10% preflop + 10% postflop for test
    # 90% preflop + 90% postflop for train
    preflop_train, preflop_test = train_test_split(
        processed_preflop_df, test_size=0.1, random_state=args.random_state
    )
    postflop_train, postflop_test = train_test_split(
        processed_postflop_df, test_size=0.1, random_state=args.random_state
    )

    # union train and test
    train_df = pd.concat([preflop_train, postflop_train], axis=0)
    test_df = pd.concat([preflop_test, postflop_test], axis=0)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_test_path), exist_ok=True)
    train_df.to_csv(args.output_train_path, index=False)
    test_df.to_csv(args.output_test_path, index=False)
