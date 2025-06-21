import pandas as pd
import torch
from torch.utils.data import Dataset

from src.vocab.poker_vocab import action_vocab, encode_action, encode_card, encode_round


class PokerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, max_action_len: int):
        """
        Args:
            dataframe (pd.DataFrame): The parsed CSV data.
            max_action_len (int): Maximum length of the action sequence. Sequences are padded if shorter.
        """
        self.samples = []
        self.max_action_len = max_action_len
        self.token_type_map = {
            "round": 0,
            "card": 1,
            "action": 2,
        }

        for _, row in dataframe.iterrows():
            input_seq = []
            type_seq = []

            # Encode round
            input_seq.append(encode_round(row["round"]))
            type_seq.append(self.token_type_map["round"])

            # Encode hole + community cards
            card_columns = [
                "hole1",
                "hole2",
                "flop1",
                "flop2",
                "flop3",
                "turn",
                "river",
            ]
            for col in card_columns:
                input_seq.append(encode_card(row[col]))
                type_seq.append(self.token_type_map["card"])

            # Encode action sequence
            action_ids = [encode_action(a) for a in row["action_sequence"]]

            # Pad action_ids
            pad_len = self.max_action_len - len(action_ids)
            action_ids += [action_vocab["empty"]] * pad_len

            # Combine full input
            input_seq.extend(action_ids)
            type_seq.extend([self.token_type_map["action"]] * len(action_ids))

            # Target: next player action
            target_action = encode_action(row["player_action"])

            self.samples.append((input_seq, type_seq, target_action))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, type_seq, target = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(type_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )
