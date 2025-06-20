# Poker Action Transformer

This project aims to build a Transformer-based neural network model that predicts the next action of a target player in a simplified Texas Hold'em poker game scenario. The model takes as input a combination of structured (card state) and sequential (betting history) information and outputs a classification among three possible actions: **Fold (F)**, **Check/Call (C)**, or **Raise (R)**.

---

## Problem Overview

The goal is to predict the **next action** of a target player given the current game state, which consists of:

- **Hole Cards**: Two face-down cards held by the player.
- **Board Cards**: Shared community cards revealed progressively in the flop, turn, and river rounds.
- **Betting History**: A sequence of player actions (F/C/R) across four betting rounds.

---

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
- **Output**: The player’s action at that point (`F`, `C`, or `R`).

---

## Project Status

This repository is currently in the **planning phase**. No code has been implemented yet.
