"""
Implementing LLM judges for comparing predicted outputs against gold outputs in pairs.
"""

from typing import List, Tuple, Mapping, Callable
from torch.utils.data import DataLoader

from judgement.data.common import utils
from judgement.eval.model_graded.judge import LLMJudge, MixtureofJudges
from judgement import * 
from judgement.constants import *
from judgement.data.common import dataset 

class ComparisonEvaluator(LLMJudge):
    
    """
    A single foundation model for judging the quality of predicted/gold output pairs.
    """
    
    def evaluate_sample(self, criteria: str, pred: str, gold: str) -> str:
        """
        Produces an evaluation of the predicted output against the gold output
        based on the judge's criteria.

        Args:
            criteria (str): the criteria for evaluation
            pred (str): the predicted output
            gold (str): the gold output
            
        """
        # 1. generate the prompt for the evaluation task
        compiled_eval_prompt = self.eval_prompt_skeleton.compile(
            criteria=criteria,
            pred=pred,
            gold=gold,
        )  # fill in the eval prompt with the dynamic vars
        # 2. run the judge model on the prompt
        chat_completion = utils.get_chat_completion(self.judge, compiled_eval_prompt)
        return chat_completion

    def evaluate_samples_batch(self, criteria, preds: List[str], golds: List[str]) -> List[str]:
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            criteria (str): the criteria for evaluation
            preds (List[str]): the predicted outputs
            golds (List[str]): the gold outputs

        Returns:
            List[str]: the responses from the judge model, one per input pair of (pred, gold)
        """
        assert len(preds) == len(golds), "Number of predictions and golds must match"
        
        compiled_eval_prompts = [self.eval_prompt_skeleton.compile(
            criteria=criteria,
            pred=pred,
            gold=gold,
        ) for pred, gold in zip(preds, golds)]  # fill in the eval prompts with the dynamic vars
        
        chat_completions = utils.get_chat_completion(self.judge, compiled_eval_prompts, batched=True)
        return chat_completions

    def evaluate_test_set(self, criteria: str, dataset: dataset.ResponseDataset, collate_fn: Callable, batch_size: int = 8) -> List[str]:
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


class ComparisonMixture(MixtureofJudges):

    """
    A mixture of multiple LLM as judges for evaluating the quality of predicted/gold output pairs.
    """

    def build_dynamic_mixture_prompt(self, judge_responses: List[str]) -> str:
        """
        Builds a dynamic prompt for mixing the judge responses, depending on the number of judges.

        Args:
            judge_responses (List[str]): the responses from the judges
        """

        
        # Format the judge responses first
        formatted_responses = "\n".join([f"# Judge {i + 1}'s response: #\n{response}" for i, response in enumerate(judge_responses)])
        """
        You are tasked with synthesizing responses from multiple expert judges. You will receive N individual answers on the same topic. Your job is to:

        1. Analyze and compare the key points, patterns, and agreements between the answers.
        2. Identify the consensus by focusing on areas where most or all of the answers align. Consider common reasoning and frequently mentioned conclusions.
        3. Condense the responses into a single, coherent, and concise answer that represents the collective judgment of the group.
        4. When opinions differ or contradict, highlight the most supported viewpoint while briefly acknowledging the dissenting perspectives.
        5. Ensure the final answer is balanced and clear, providing a comprehensive summary that captures the wisdom of all judges while avoiding repetition.

        ## Start of Judge Responses ##
        {{judge_responses}}
        ## End of Judge Responses ##
        Synthesized response:
        """
        # Inject the judge responses into the mixture prompt
        compiled_mixture_prompt = self.mixture_base_prompt.compile(
            judge_responses=formatted_responses,
        )
        return compiled_mixture_prompt
        
    def evaluate_sample(self, pred: str, gold: str, criteria: str):
        """
        Produces an evaluation of the predicted output against the gold output based on the judges' criteria.

        Args:
            pred (str): the predicted output
            gold (str): the gold output
            criteria (str): the criteria for evaluation
        """
        # Create evaluation prompt 
        compiled_eval_prompt = self.eval_prompt_skeleton.compile(
            criteria=criteria,
            pred=pred,
            gold=gold,
            judges=self.judges,
        )

        # Collect all judge responses 
        responses = utils.get_completion_multiple_models(
            models=self.judges,
            messages=[compiled_eval_prompt] * len(self.judges),
        )

        # Compile responses into the mixture prompt
        compiled_mixture_prompt = self.build_dynamic_mixture_prompt(responses)
        mixed_response = utils.get_chat_completion(
            model_type=self.aggregator,
            messages=compiled_mixture_prompt,
        )
        return mixed_response

    def evaluate_samples_batch(self, preds: List[str], golds: List[str], criteria: str) -> List[str]:
        """
        Produces an evaluation of the predicted outputs against the gold outputs in the dataset.

        Args:
            preds (List[str]): the predicted outputs
            golds (List[str]): the gold outputs
            criteria (str): the criteria for evaluation
        """
        return [self.evaluate_sample(pred, gold, criteria) for pred, gold in zip(preds, golds)]    

    def evaluate_test_set(self, criteria: str, dataset: dataset.ResponseDataset, collate_fn: Callable, batch_size: int) -> List[str]:
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
