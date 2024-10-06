"""
LLM as a judge general implementation

Assume you have:
- dataset: pairs of (predicted, gold) outputs for a task
- a foundation model
- a criteria for judging the quality of the predicted/gold outputs 
"""

from typing import List, Tuple, Mapping, Callable
from judgement.data.common import dataset 


class LLMJudge:
    
    """
    A foundation model for judging the quality of predicted/gold outputs
    """

    def __init__(self, judge):
        self.judge = judge        
        self.eval_prompt = ""  # base prompt for the evaluation task  TODO write this
    
    def evaluate_sample(self, criteria: str, pred: str, gold: str):
        """
        Produces an evaluation of the predicted output against the gold output
        based on the judge's criteria.

        Args:
            criteria (str): the criteria for evaluation
            pred (str): the predicted output
            gold (str): the gold output
            
        """
        # TODO 
        # 1. generate the prompt for the evaluation task
        # 2. run the judge model on the prompt
        pass 

    def evaluate_test_set(self, criteria: str, dataset: dataset.ResponseDataset, collate_fn: Callable, batch_size: int):
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            criteria (str): the criteria for evaluation
            dataset (dataset.ResponseDataset): the dataset containing the predicted and gold outputs
            collate_fn (Callable): the collate function to use for batching the data
            batch_size (int): the batch size to use for evaluation
        """
        pass


class MixtureofJudges:

    """
    A mixture of multiple LLM as judges
    """

    def __init__(self, judges: List[str], mixture_prompt: str):
        self.judges = [LLMJudge(model) for model in judges]
        self.eval_prompt = ""  # base prompt for the evaluation task  TODO write this
        self.mixture_prompt = mixture_prompt

    def evaluate_sample(self, pred: str, gold: str, criteria: str):
        """
        Produces an evaluation of the predicted output against the gold output based on the judges' criteria.

        Args:
            pred (str): the predicted output
            gold (str): the gold output
            criteria (str): the criteria for evaluation
        """
        # TODO
        # 1. generate the prompt for the evaluation task
        # 2. run the judge models on the prompt (parallelized)
        # 3. aggregate the results
        pass

    def evaluate_test_set(self, criteria: str, dataset: dataset.ResponseDataset, collate_fn: Callable, batch_size: int):
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            criteria (str): the criteria for evaluation
            dataset (dataset.ResponseDataset): the dataset containing the predicted and gold outputs
            collate_fn (Callable): the collate function to use for batching the data
            batch_size (int): the batch size to use for evaluation
        """
        pass

