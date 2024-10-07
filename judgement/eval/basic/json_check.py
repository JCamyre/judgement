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

    def evaluate_sample(self, output: Union[str, dict]) -> bool:
        """
        Produces an evaluation of a single predicted output

        Args:
            output (Union[str, dict]): the predicted output. Can be a JSON string or a loaded JSON object (dict)
        """

        if type(output) == str:
            return is_valid_json(output)
        elif type(output) == dict:
            return is_valid_json_dict(output)
        else:
            raise TypeError(f"Output must be a JSON string or a loaded JSON object. Got: {type(output)}")

    def evaluate_all_samples(self, outputs: List[str]) -> List[str]:
        """
        Produces an evaluation of multiple predicted outputs

        TODO implement this
        """
        pass 

    def run(self):
        """
        Runs the evaluation

        TODO implement this
        """
        pass
