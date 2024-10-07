"""
LLM as a judge general implementation for evaluating the quality of predicted outputs 
against gold outputs according to a given criteria.

Assume you have:
- dataset: pairs of (predicted, gold) outputs for a task
- a foundation model
- a criteria for judging the quality of the predicted/gold outputs 
"""

from abc import ABC, abstractmethod
import langfuse as lf
from typing import List, Tuple, Mapping, Callable
from judgement import * 
from judgement.constants import *
from judgement.litellm_model_names import LITE_LLM_MODEL_NAMES
from judgement.eval.model_graded.exceptions import JudgeNotSupportedError


class LLMJudge(ABC):
    """
    An abstract class for judging the quality of outputs using an LLM
    """

    def __init__(self, judge: str, eval_prompt_skeleton: lf.client.ChatPromptClient):
        """
        Initializes the LLMJudge with the given judge model and evaluation prompt.

        Args:
            judge (str): Model name for the judge
            eval_prompt (lf.client.ChatPromptClient): Base prompt for the evaluation task; Langfuse object that holds the base prompt and can be compiled with dynamic args.
        """
        if judge not in LITE_LLM_MODEL_NAMES and judge not in TOGETHER_SUPPORTED_MODELS:
            raise JudgeNotSupportedError(judge)
        self.judge = judge 

        if type(eval_prompt_skeleton) == str:
            raise TypeError(f"Eval prompt must be a Langfuse chat prompt object to compile args. Got: {type(eval_prompt_skeleton)}")       
        self.eval_prompt_skeleton = eval_prompt_skeleton  # base prompt for the evaluation task

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> str:
        """
        Produces an evaluation of a single predicted output
        """  
        pass 

    @abstractmethod
    def evaluate_samples_batch(self, *args, **kwargs) -> List[str]:
        """
        Produces an evaluation of multiple predicted outputs
        """
        pass

    @abstractmethod
    def evaluate_test_set(self, *args, **kwargs) -> List[str]:
        """
        Produces an evaluation across a whole dataset.
        """
        pass


class MixtureofJudges(ABC):
    
    """
    A mixture of multiple LLM as judges
    """

    def __init__(self, judges: List[str], aggregator: str, eval_prompt_skeleton: lf.client.ChatPromptClient, mixture_base_prompt: lf.client.ChatPromptClient):
        for judge in judges:
            if judge not in LITE_LLM_MODEL_NAMES and judge not in TOGETHER_SUPPORTED_MODELS:
                raise JudgeNotSupportedError(judge)
        self.judges = judges  # list of judge model names

        if aggregator not in LITE_LLM_MODEL_NAMES and aggregator not in TOGETHER_SUPPORTED_MODELS:
            raise JudgeNotSupportedError(aggregator)
        self.aggregator = aggregator  # model name for the aggregator judge
        self.eval_prompt_skeleton = eval_prompt_skeleton  # base prompt for the evaluation task 
        self.mixture_base_prompt = mixture_base_prompt  # prompt for mixing judge answers

    @abstractmethod
    def build_dynamic_mixture_prompt(self, *args, **kwargs) -> str:
        """
        Builds a dynamic prompt for mixing the judge responses, depending on the number of judges.
        """
        pass

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> str:
        """
        Produces an evaluation of task output based on the judges' criteria.
        """
        pass

    @abstractmethod
    def evaluate_samples_batch(self, *args, **kwargs) -> List[str]:
        """
        Produces an evaluation of samples across a batch of data.
        """
        pass

    @abstractmethod
    def evaluate_test_set(self, *args, **kwargs) -> List[str]:
        """
        Produces an evaluation of samples across the entire dataset..
        """
        pass
