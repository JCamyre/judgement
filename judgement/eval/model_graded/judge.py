"""
LLM as a judge general implementation

Assume you have:
- dataset: pairs of (predicted, gold) outputs for a task
- a foundation model
- a criteria for judging the quality of the predicted/gold outputs 
"""

from typing import List, Tuple, Mapping


class LLMJudge:
    
    """
    A foundation model for judging the quality of predicted/gold outputs
    """

    def __init__(self, judge):
        self.judge = judge        
        self.eval_prompt = ""  # prompt for the evaluation task  TODO write this
    
    def evaluate_sample(self, pred, gold, criteria):
        """
        Produces an evaluation of the predicted output against the gold output
        based on the judge's criteria.
        """
        pass 

    def evaluate_test_set(self, dataset):
        pass


class MixtureofJudges:

    """
    A mixture of multiple LLM as judges
    """

    def __init__(self, judges: List[str], mixture_prompt: str):
        self.judges = [LLMJudge(model) for model in judges]
        self.mixture_prompt = mixture_prompt

    def evaluate_sample(self, pred, gold):
        pass

    def evaluate_test_set(self, dataset):
        pass

