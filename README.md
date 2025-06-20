# Poker Action Transformer

This project aims to build a Transformer-based neural network model that predicts the next action of a target player in a simplified Texas Hold'em poker game scenario. The model takes as input a combination of structured (card state) and sequential (betting history) information and outputs a classification among three possible actions: **Fold (F)**, **Check/Call (C)**, or **Raise (R)**.

## Problem Overview

The goal is to predict the **next action** of a target player given the current game state, which consists of:

- **Hole Cards**: Two face-down cards held by the player.
- **Board Cards**: Shared community cards revealed progressively in the flop, turn, and river rounds.
- **Betting History**: A sequence of player actions (F/C/R) across four betting rounds.

## Input Encoding Strategy

### Card Encoding

- All cards are represented as categorical tokens (52 real cards + 1 special token for "empty").
- A total of **53 tokens** are used for card embedding.
- Embeddings for cards are learned.

**Note**: Hole cards and flop cards have **no inherent order**. Turn, River, and betting sequences are **order-sensitive**.

### Betting History

- Represented as a token sequence with possible values {F, C, R}.
- Separated by round (Preflop / Flop / Turn / River).
- Each action token is embedded.
- Treated as a standard sequential input to the Transformer.

---

## Label Definition

Each training example corresponds to a decision point in the game:

- **Input**: All game information *up to* the current point.
- **Output**: The player's action at that point (`F`, `C`, or `R`).

## Data Processing

The project includes a complete data processing pipeline that converts raw PokerBench dataset files into a standardized format suitable for training.

### Dataset Structure

The processed dataset contains the following columns:
- `round`: Game round (preflop, flop, turn, river)
- `hole1`, `hole2`: Player's hole cards (e.g., "As", "Kd")
- `flop1`, `flop2`, `flop3`: Community flop cards
- `turn`: Turn card (or "empty" if not applicable)
- `river`: River card (or "empty" if not applicable)
- `action_sequence`: Betting history as F/C/R sequence
- `player_action`: Target action to predict (F/C/R)

### Running Data Preprocessing

To process the raw datasets and generate train/test splits:

```bash
PYTHONPATH=. python3 scripts/preprocess_data.py \
    --preflop_input_path dataset/raw/preflop_1k_test_set_game_scenario_information.csv \
    --postflop_input_path dataset/raw/postflop_10k_test_set_game_scenario_information.csv \
    --output_train_path dataset/processed/train.csv \
    --output_test_path dataset/processed/test.csv
```

This will:
- Process both preflop and postflop datasets
- Convert actions to standardized F/C/R format
- Split data into 90% training and 10% test sets
- Save processed files to `dataset/processed/`

### Data Sources

The project uses the PokerBench dataset, which contains:
- **Preflop data**: 1,000 test scenarios with optimal decisions
- **Postflop data**: 10,000 test scenarios across flop, turn, and river rounds

For detailed dataset information, see `dataset/README.md`.

## Project Status

This repository is currently in the **implementation phase**. 

### Completed
- [x] Raw dataset integration
- [x] Data processing pipeline

### Planned
- [ ] Transformer model architecture
- [ ] Training pipeline
- [ ] Model evaluation
- [ ] Documentation
