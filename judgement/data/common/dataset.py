"""
This code implements the dataset class for the judgement library
"""

from torch.utils.data import Dataset, DataLoader
from typing import List, Mapping, Tuple 

# Dataset class is a container for the data used in the evaluation
class ResponseDataset(Dataset):

    """
    Dataset that holds pairs of (predicted, gold) response (strings) for a task.
    """

    def __init__(self, predicted_texts: List[str], gold_texts: List[str]):
        """
        Args:
            predicted_texts: list of predicted texts
            gold_texts: list of gold texts
        """
        assert len(predicted_texts) == len(gold_texts), "Both lists must have the same length."
        self.predicted_texts = predicted_texts
        self.gold_texts = gold_texts

    def __len__(self):
        """Returns the total number of text pairs in the dataset."""
        return len(self.predicted_texts)

    def __getitem__(self, idx: int):
        """Retrieves a pair (predicted, gold) at the given index."""
        return self.predicted_texts[idx], self.gold_texts[idx]


def response_pair_collate_fn(batch: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    Collates a batch of (predicted, gold) pairs into a tuple of lists.
    Args:
        batch: List of tuples (predicted, gold) texts.
    
    Returns:
        Tuple of lists: ([predicted_texts], [gold_texts])
    """
    predicted_texts, gold_texts = zip(*batch)
    return list(predicted_texts), list(gold_texts)


if __name__ == "__main__":
    # Example usage of the ResponseDataset and response_pair_collate_fn
    predicted_texts = ["Pred 1", "Pred 2", "Pred 3"]
    gold_texts = ["Gold 1", "Gold 2", "Gold 3"]

    # Create the dataset
    string_pair_dataset = ResponseDataset(predicted_texts, gold_texts)

    # Create the DataLoader with a custom collate_fn
    string_pair_loader = DataLoader(string_pair_dataset, batch_size=4, collate_fn=response_pair_collate_fn)

    # Iterate through the dataloader
    for predicted_batch, gold_batch in string_pair_loader:
        print("Predicted:", predicted_batch)
        print("Gold:", gold_batch)
        print("-" * 30)  

