# Poker Action Transformer

A PyTorch-based Transformer neural network that predicts the next action (Fold/Call/Raise) for a target player in Texas Hold'em poker games.

## Problem Statement

Given a poker game state, predict the optimal next action for a target player. The model takes three key inputs:

- **Hole Cards**: Player's two face-down cards (e.g., `7h5c`, `3d4s`)
- **Board Cards**: Community cards revealed in each round (e.g., `7dAcJh/2d/Td`)
- **Bet History**: Sequence of actions across 4 betting rounds (e.g., `FFFRFC/CC/CC/CR`)

**Output**: One of three actions - **F** (Fold), **C** (Call/Check), or **R** (Raise)

## Architecture

### Model Design
- **Transformer Encoder**: Multi-head attention mechanism for sequence modeling
- **Input Encoding**: 
  - 53 card tokens (52 cards + 1 empty token)
  - 5 action tokens (F/C/R + separator + padding)
  - 4 round tokens (preflop/flop/turn/river)
- **Output**: 3-class classification (F/C/R)

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
PYTHONPATH=. python3 scripts/preprocess_data.py \
    --preflop_input_path dataset/raw/preflop_1k_test_set_game_scenario_information.csv \
    --postflop_input_path dataset/raw/postflop_10k_test_set_game_scenario_information.csv \
    --output_train_path dataset/processed/train.csv \
    --output_test_path dataset/processed/test.csv
```

### 3. Train Model
```bash
PYTHONPATH=. python3 scripts/train_poker_model.py
```

## Project Structure

```
poker-action-transformer/
├── src/
│   ├── models/poker_transformer.py    # Transformer model implementation
│   ├── data/
│   │   ├── poker_dataset.py          # Dataset class
│   │   └── preprocessing.py          # Data preprocessing
│   └── vocab/poker_vocab.py          # Vocabulary and encoding
├── scripts/
│   ├── train_poker_model.py          # Training script
│   └── preprocess_data.py            # Data preprocessing script
└── dataset/
    ├── raw/                          # Original PokerBench data
    └── processed/                    # Processed train/test splits
```

## Data Format

### Input Example
```json
{
    "round": "river",
    "hole1": "As",           # Ace of spades
    "hole2": "Kh",           # King of hearts
    "flop1": "7d",           # 7 of diamonds
    "flop2": "Ac",           # Ace of clubs
    "flop3": "Jh",           # Jack of hearts
    "turn": "2d",            # 2 of diamonds
    "river": "7h",           # 7 of hearts 
    "action_sequence": "FFFRFC/CC/CC/CR"  # Betting history
}
```

### Output
```python
"R"  # Model predicts: Raise
```
