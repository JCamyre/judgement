"""
Checks that the output is valid JSON format
"""

import json 
from typing import Union, List, Mapping
from judgement.eval.basic.basic_eval import BasicEval


def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except ValueError:
        return False
    
def is_valid_json_dict(s: Mapping) -> bool:
    try:
        json.dumps(s)
        return True
    except (TypeError, ValueError):
        return False


class JsonValidator(BasicEval):

    def evaluate_sample(self, sample: Union[str, dict]) -> bool:
        """
        Produces an evaluation of a single sample JSON 

        Args:
            sample (Union[str, dict]): the sample to evaluate. Can be a JSON string or a loaded JSON object (dict)
        """
        if type(sample) == str:
            return is_valid_json(sample)
        elif type(sample) == dict:
            return is_valid_json_dict(sample)
        else:
            raise TypeError(f"Output must be a JSON string or a loaded JSON object. Got: {type(sample)}")

    def run(self):
        """
        Runs the evaluation
        """
        # Load the dataset
        # Evaluate all samples in the dataset
        # Save the results
        pass
