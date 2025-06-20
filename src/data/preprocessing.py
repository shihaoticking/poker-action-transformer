import numpy as np
import pandas as pd


def parse_preflop_decision(decision: str) -> str:
    if decision in ('call', 'check'):
        return 'C'
    elif decision.endswith('bb') or decision == 'allin':
        return 'R'
    elif decision == 'fold':
        return 'F'
    else:
        raise ValueError(f"Invalid action: {decision}")


def parse_preflop_action_sequence(prev_line: str) -> str:
    """
    Parse the action sequence from the previous line.

    Input:
        prev_line: str, the previous line of the game sequence. Formatted as {POSITION}/{ACTION}.

    Return:
        action_sequence: str, the action sequence. Formatted as {ACTION}.
            - R: bet or raise
            - C: call
            - F: fold

    Example 1:

        Input: 'UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/BTN/fold'
        Return: 'RCRRFF'
    
    Example 2:

        Input: 'SB/call'
        Return: 'C'

    Example 3:

        Input: 'HJ/2.0bb/CO/call/BTN/call'
        Return: 'RCC'
    """
    actions = prev_line.split('/')
    action_sequence = ''
    for i in range(1, len(actions), 2):
        action_sequence += parse_preflop_decision(actions[i])

    return action_sequence


def process_preflop_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the preflop dataset.

    Return:
        df: pd.DataFrame, the processed dataset with the following columns:
            - round: str, the round of the game. fixed to 'preflop'
            - hole1: str, the first hole card of the player.
            - hole2: str, the second hole card of the player.
            - flop1: str, the first flop card. fixed to 'empty'
            - flop2: str, the second flop card. fixed to 'empty'
            - flop3: str, the third flop card. fixed to 'empty'
            - turn: str, the turn card. fixed to 'empty'
            - river: str, the river card. fixed to 'empty'
            - action_sequence: str, the action sequence. Formatted as {ACTION}{ACTION}...{ACTION}.
                - R: bet or raise
                - C: call
                - F: fold
            - player_action: str, the action of the player.
                - R: bet or raise
                - C: call
                - F: fold
    """
    processed_df = pd.DataFrame({
        'round': 'preflop',
        'hole1': df['hero_holding'].str[0:2],
        'hole2': df['hero_holding'].str[2:4],
        'flop1': 'empty',
        'flop2': 'empty',
        'flop3': 'empty',
        'turn': 'empty',
        'river': 'empty',
        'action_sequence': df['prev_line'].apply(parse_preflop_action_sequence),
        'player_action': df['correct_decision'].apply(parse_preflop_decision)
    })
    
    return processed_df


def parse_postflop_action_sequence(postflop_action: str) -> str:
    """
    Parse the action sequence from the postflop action.

    Input:
        postflop_action: str, the postflop action. Formatted as {POSITION}_{ACTION}/{POSITION}_{ACTION}/.../{POSITION}_{ACTION}.
        or dealcards/{CARD} in the middle of the action sequence.

    Return:
        action_sequence: str, the action sequence. Formatted as {ACTION}{ACTION}...{ACTION}.
            - R: bet or raise
            - C: call
            - F: fold

    Example 1:
        Input: 'OOP_CHECK/IP_CHECK/dealcards/Jc/OOP_CHECK/IP_BET_5/OOP_RAISE_14/IP_CALL/dealcards/7c/OOP_CHECK'
        Return: 'CC/CRRC/C'
    
    Example 2:
        Input: 'OOP_BET_2/IP_RAISE_8/OOP_CALL/dealcards/3c/OOP_CHECK'
        Return: 'RRC/C'

    Example 3:
        Input: 'OOP_CHECK/IP_CHECK/dealcards/2d/OOP_CHECK/IP_BET_6/OOP_RAISE_17/IP_CALL/dealcards/Ac/OOP_BET_20/IP_RAISE_80'
        Return: 'CC/CRRC/RR'
    """
    actions = postflop_action.split('/')
    action_sequence = ''
    for action in actions:
        if action == 'dealcards':
            action_sequence += '/'
        elif action.endswith('CHECK') or action.endswith('CALL'):
            action_sequence += 'C'
        elif 'BET' in action or 'RAISE' in action:
            action_sequence += 'R'

    return action_sequence


def parse_postflop_decision(decision: str) -> str:
    if decision == 'Fold':
        return 'F'
    elif decision in ('Call', 'Check'):
        return 'C'
    elif decision.startswith('Raise') or decision.startswith('Bet'):
        return 'R'
    else:
        raise ValueError(f"Invalid decision: {decision}")


def process_postflop_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the postflop dataset.

    Return:
        df: pd.DataFrame, the processed dataset with the following columns:
            - round: str, the round of the game.
            - hole1: str, the first hole card of the player.
            - hole2: str, the second hole card of the player.
            - flop1: str, the first flop card.
            - flop2: str, the second flop card.
            - flop3: str, the third flop card.
            - turn: str, the turn card.
            - river: str, the river card.
            - action_sequence: str, the action sequence. Formatted as {ACTION}{ACTION}...{ACTION}.
                - R: bet or raise
                - C: call
                - F: fold
            - player_action: str, the action of the player.
                - R: bet or raise
                - C: call
                - F: fold
    """
    processed_df = pd.DataFrame({
        'round': df['evaluation_at'].str.lower(),
        'hole1': df['holding'].str[0:2],
        'hole2': df['holding'].str[2:4],
        'flop1': df['board_flop'].str[0:2],
        'flop2': df['board_flop'].str[2:4],
        'flop3': df['board_flop'].str[4:6],
        'turn': np.where(df['evaluation_at'].str.lower() == 'flop', 'empty', df['board_turn']),
        'river': np.where(df['evaluation_at'].str.lower() == 'river', df['board_river'], 'empty'),
        'action_sequence': df['preflop_action'].apply(parse_preflop_action_sequence) + '/' + df['postflop_action'].apply(parse_postflop_action_sequence),
        'player_action': df['correct_decision'].apply(parse_postflop_decision)
    })
    
    return processed_df
