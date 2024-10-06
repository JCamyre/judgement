"""
LLM as a judge general implementation for evaluating the quality of predicted outputs 
against gold outputs according to a given criteria.

Assume you have:
- dataset: pairs of (predicted, gold) outputs for a task
- a foundation model
- a criteria for judging the quality of the predicted/gold outputs 
"""

import langfuse as lf
from torch.utils.data import DataLoader
from typing import List, Tuple, Mapping, Callable
from judgement import * 
from judgement.data.common import dataset 
from judgement.data.cleaning import utils


class LLMJudge:
    
    """
    A foundation model for judging the quality of predicted/gold outputs
    """

    def __init__(self, judge: str, eval_prompt_skeleton: lf.client.ChatPromptClient):
        """
        Initializes the LLMJudge with the given judge model and evaluation prompt.

        Args:
            judge (str): Model name for the judge
            eval_prompt (lf.client.ChatPromptClient): Base prompt for the evaluation task; Langfuse object that holds the base prompt and can be compiled with dynamic args.
        """
        self.judge = judge  # TODO: check that this is a valid model name

        if type(eval_prompt_skeleton) == str:
            raise TypeError(f"Eval prompt must be a Langfuse chat prompt object to compile args. Got: {type(eval_prompt_skeleton)}")       
        self.eval_prompt_skeleton = eval_prompt_skeleton  # base prompt for the evaluation task  
    
    def evaluate_sample(self, criteria: str, pred: str, gold: str) -> str:
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
        compiled_eval_prompt = self.eval_prompt_skeleton.compile(
            criteria=criteria,
            pred=pred,
            gold=gold,
        )  # fill in the eval prompt with the dynamic vars
        # 2. run the judge model on the prompt
        chat_completion = utils.get_chat_completion(self.judge, compiled_eval_prompt)

        # 3. extract the evaluation from the completion  TODO
        return chat_completion

    def evaluate_samples_batch(self, criteria, preds: List[str], golds: List[str]) -> List[str]:
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            criteria (str): the criteria for evaluation
            preds (List[str]): the predicted outputs
            golds (List[str]): the gold outputs
        """
        compiled_eval_prompts = [self.eval_prompt_skeleton.compile(
            criteria=criteria,
            pred=pred,
            gold=gold,
        ) for pred, gold in zip(preds, golds)]  # fill in the eval prompts with the dynamic vars
        
        chat_completions = utils.get_chat_completion(self.judge, compiled_eval_prompts, batched=True)
        return chat_completions


    def evaluate_test_set(self, criteria: str, dataset: dataset.ResponseDataset, collate_fn: Callable, batch_size: int = 8):
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            criteria (str): the criteria for evaluation
            dataset (dataset.ResponseDataset): the dataset containing the predicted and gold outputs
            collate_fn (Callable): the collate function to use for batching the data
            batch_size (int): the batch size to use for evaluation
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        total_responses = []
        for predicted_batch, gold_batch in dataloader:
            responses = self.evaluate_samples_batch(criteria, predicted_batch, gold_batch)
            total_responses.extend(responses)
        return total_responses


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

