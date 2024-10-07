"""
Implements the base class of a basic evaluation 
"""

from abc import ABC, abstractmethod
from typing import List


class BasicEval(ABC):
    """
    Non-deterministic evaluation of a model's output

    When implementing a basic evaluation, the following methods must be implemented:
    `evaluate_sample`: Produces an evaluation of a single predicted output
    `run`: Takes a recorder and runs the evaluation. Typically, most `run` methods will follow the same pattern:
    loading the data, calling `evaluate_all_samples`, and then saving the aggregated results.`
    """

    def __init__(self, eval_prompt_skeleton: str):
        self.eval_prompt_skeleton = eval_prompt_skeleton  # base prompt for the evaluation task

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> str:
        """
        Produces an evaluation of a single predicted output
        """
        pass

    @abstractmethod 
    def evaluate_all_samples(self, *args, **kwargs) -> List[str]:
        """
        Produces an evaluation of multiple predicted outputs
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Runs the evaluation
        """
        pass
