# PokerBench Dataset Overview

The dataset is downloaded from PokerBench Dataset contains game scenarios and optimal decisions computed by solvers in No Limit Texas Hold'em. It is divided into pre-flop and post-flop datasets, each with training and test splits. We only downloaded the test dataset here. The data is stored in .csv formats:

- CSV files: Contain structured game information from which the JSON files were generated. The pre-flop and post-flop CSV files have different structures to accommodate the different stages of the game.

# Dataset Structure

## CSV Files

The CSV files store structured game scenario information. They include details of player actions, positions, and board state. The structure of the columns differs for pre-flop and post-flop datasets.

### Pre-Flop CSV

Columns:
1. prev_line: The sequence of player actions before the current decision point, formatted as {Position}/{Action}/{Amount}. E.g., UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin.
 2.	hero_pos: The position of the player making the decision (UTG, HJ, CO, BTN, SB, or BB).
 3.	hero_holding: The player's hole cards (e.g., KdKc for King of Diamonds and King of Clubs).
 4.	correct_decision: The optimal decision for the player (call, fold, etc.).
 5.	num_players: The number of players still in the hand at the decision point.
 6.	num_bets: The number of betting rounds/actions that have occurred so far.
 7.	available_moves: The possible decisions the player can make (e.g., ['call', 'fold']).
 8.	pot_size: The current size of the pot at the decision point.

#### Example Row:

UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold, SB, KdKc, call, 4, 3, ['call', 'fold'], 117.0

### Post-Flop CSV

Columns: 
1. preflop_action: The sequence of player actions leading to the flop, formatted as {Position}/{Action}/{Amount}.
 2.	board_flop: The three community cards on the flop (e.g., Ks7h2d).
 3.	board_turn: The turn card, if available (e.g., Jc).
 4.	board_river: The river card, if available (e.g., 7c).
 5.	aggressor_position: The position of the most recent aggressor in the hand (OOP for out of position, IP for in position).
 6.	postflop_action: The sequence of player actions post-flop, formatted as {Position}\_{Action}\/{Position}\_{Action}. E.g., OOP_CHECK/IP_BET_5/OOP_RAISE_14.
 7.	evaluation_at: The street at which the decision is evaluated (Flop, Turn, or River).
 8.	available_moves: The possible decisions the player can make (e.g., ['Check', 'Bet 24']).
 9.	pot_size: The current size of the pot at the decision point.
 10. hero_position: The position of the player making the decision (UTG, HJ, CO, BTN, SB, or BB).
 11. holding: The player's hole cards (e.g., 8h8c for two eights of hearts and clubs).
 12. correct_decision: The optimal decision for the player (Check, Call, Bet, etc.).

#### Example Row:

HJ/2.0bb/BB/call, Ks7h2d, Jc, 7c, OOP, OOP_CHECK/IP_CHECK/dealcards/Jc/OOP_CHECK/IP_BET_5/OOP_RAISE_14, River, ['Check', 'Bet 24'], 32, IP, 8h8c, Check

## File Descriptions

| Dataset Type | File Name | Description |
|--------------|-----------|-------------|
| Pre-Flop Dataset | `preflop_1k_test_set_game_scenario_information.csv` | Structured game information for 1,000 test examples |
| Post-Flop Dataset | `postflop_10k_test_set_game_scenario_information.csv` | Structured game information for 10,000 test examples |
