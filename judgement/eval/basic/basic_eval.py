"""
Implements the base class of a basic evaluation 
"""

from abc import ABC, abstractmethod
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
from judgement.constants import MAX_WORKER_THREADS


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
    def run(self, *args, **kwargs):
        """
        Runs the evaluation
        """
        pass
    
    def evaluate_all_samples(self, samples: List[Any]) -> List[Any]:
        """
        Produces an evaluation of multiple samples

        Args:
            samples (List[Any]): List of samples to evaluate

        Returns:
            List[Any]: List of outputs from the evaluation in the same order as the input samples

        If an output is None, there was an error during thread execution
        """
        # Get the number of worker threads from the environment variable
        num_workers = int(os.getenv('NUM_WORKER_THREADS', MAX_WORKER_THREADS))  

        # Initialize results to maintain ordered outputs
        results = [None] * len(samples)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks to the executor with their index
            futures = {executor.submit(self.evaluate_sample, sample): idx for idx, sample in enumerate(samples)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Handle exceptions raised during thread execution
                    print(f"An error occurred: {e}")
                    results[idx] = None  # Append None or handle as needed
        
        return results

