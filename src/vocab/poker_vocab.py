# Round vocabulary
round_vocab = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Suits and ranks
ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
suits = ["c", "d", "h", "s"]

# Generate 52 cards + "empty"
all_cards = [rank + suit for rank in ranks for suit in suits]
card_vocab = {card: idx for idx, card in enumerate(all_cards)}
card_vocab["empty"] = len(card_vocab)  # 52nd index

# Reverse card vocab
inv_card_vocab = {v: k for k, v in card_vocab.items()}

# Actions: Call, Raise, Fold, Round separator, Padding
action_vocab = {"C": 0, "R": 1, "F": 2, "/": 3, "empty": 4}
inv_action_vocab = {v: k for k, v in action_vocab.items()}

# Utility Functions


def encode_round(round_str: str) -> int:
    return round_vocab.get(round_str, 0)


def encode_card(card_str: str) -> int:
    return card_vocab.get(card_str, card_vocab["empty"])


def decode_card(card_idx: int) -> str:
    return inv_card_vocab.get(card_idx, "empty")


def encode_action(action_str: str) -> int:
    return action_vocab.get(action_str, action_vocab["empty"])


def decode_action(action_idx: int) -> str:
    return inv_action_vocab.get(action_idx, "empty")


def get_vocab_sizes() -> dict:
    return {
        "round_vocab_size": len(round_vocab),
        "card_vocab_size": len(card_vocab),
        "action_vocab_size": len(action_vocab),
    }
