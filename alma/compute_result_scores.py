"""
Scores the output of criteria evaluation

You can see the average score per model (x/10)
You can see the average score per file (x/10)
"""

import os 
import re 
import pprint
from typing import List, Mapping, Any


### SCORING CONVERSION ###
"""
When model is biased towards first document:
    If A is final draft and B is rough draft, 
    ['[[A>>B]]', '[[A>>B]]', '[[A>B]]'] -> [5, 5, 4] -> 4.67

    If A is rough draft and B is final draft,
    ['[[A>>B]]', '[[A>>B]]', '[[A>B]]'] -> [1, 1, 2] -> 1.33

Assume model unbiased and correct:
    If A is final draft and B is rough draft, 
    ['[[A>>B]]', '[[A>>B]]', '[[A>B]]'] -> [5, 5, 4] -> 4.67

    If A is rough draft and B is final draft,
    ['[[A<<B]]', '[[A<<B]]', '[[A<B]]'] -> [5, 5, 4] -> 4.67

Assume model unbiased, incorrect
    If A is final draft and B is rough draft,
    ['[[A<B]]', '[[A<B]]', '[[A<B]]'] -> [2, 2, 2] -> 2

Key idea: Map to reward choosing the final draft over the rough draft
"""
SCORE_CONVERSION_FD_FIRST = {
    '[[A>>B]]': 5,
    '[[A>B]]': 4,
    '[[A=B]]': 3,
    '[[A<B]]': 2,
    '[[A<<B]]': 1
}  # reward model for choosing final draft over rough draft

SCORE_CONVERSION_RD_FIRST = {
    '[[A>>B]]': 1,
    '[[A>B]]': 2,
    '[[A=B]]': 3,
    '[[A<B]]': 4,
    '[[A<<B]]': 5
}  # punish model for choosing rough draft over final draft

def extract_scores(file_path: str) -> List[dict]:
    """
    Each line of the file has this structure: 
    Comparing <file_path_1> and <file_path_2> using judge MISTRAL_8x22B_INSTRUCT Scores: ['[[A>B]]', '[[A>B]]', '[[A>>B]]']

    We want to extract the scores from the file, along with the file name for those scores.

    Args:
        file_path (str): Path to the file containing the scores.
    
    Returns:
        List[dict]: List of dictionaries containing the file names and scores. Each dict has the following structure:
        {
            "rd_file": str,  # path to the rough draft file
            "fd_file": str,  # path to the final draft file
            "scores": List[str]  # list of scores from the judges, e.g. ['[[A>B]]', '[[A>B]]', '[[A>>B]]']
        }

    """
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"File {file_path} not found.")
    results = []
    pattern = re.compile(r"Comparing (.+?) and (.+?) using judges? .+? Scores: (.+)")

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = pattern.search(line)
            if match:
                rd_file = match.group(1).strip()
                fd_file = match.group(2).strip()
                scores = eval(match.group(3).strip())  # Convert string representation of list to actual list
                results.append({
                    "rd_file": rd_file,
                    "fd_file": fd_file,
                    "scores": scores
                })
    
    return results


def score_str_to_num(scores: List[str], scoring_map: dict) -> int:
    """
    Converts a list of model voting score to a single averaged score based on the scoring map.

    Args:
        scores (List[str]): List of scores from models
        scoring_map (dict): Mapping of model scores to numerical values

    Returns the average score of valid scores in the list.

    Example:

        scores = [None, '[[A>>B]]', None]
        map = {'[[A>>B]]': 5, '[[A>B]]': 4, '[[A=B]]': 3, '[[A<B]]': 2, '[[A<<B]]': 1}  
        [None, '[[A>>B]]', None] -> [NaN, 5, Nan] -> 5
    
    ['[[A>>B]]', '[[A>>B]]', '[[A>B]]'] -> [5, 5, 4] -> 4.67
    """
    result_scores = []
    for s in scores:
        if s not in scoring_map:
            continue 
        result_scores.append(scoring_map[s])
    return sum(result_scores) / len(result_scores)


def map_files(dir: str) -> dict:
    """
    Maps the model names to the file paths containing the scores.

    Returns:
        {
            "model_name1": [
                {
                    "rd_file": str,  # path to the rough draft file
                    "fd_file": str,  # path to the final draft file
                    "scores": List[str]  # list of scores from the judges, e.g. ['[[A>B]]', '[[A>B]]', '[[A>>B]]']
                },
            ],
            "model_name2": [
                {
                    "rd_file": str,  # path to the rough draft file
                    "fd_file": str,  # path to the final draft file
                    "scores": List[str]  # list of scores from the judges, e.g. ['[[A>B]]', '[[A>B]]', '[[A>>B]]']
                },
            ],
        }
    """
    res = {}
    for FILE_NAME in os.listdir(dir):
        # FILE_NAME takes the pattern of <model_name>_results.txt
        # We want to extract the model name from the file name with regex
        pattern = re.compile(r"(.+?)_results.txt")
        match = pattern.search(FILE_NAME)
        if match:
            abs_file_path = os.path.join(dir, FILE_NAME)
            model_name = match.group(1)
            # map the model name to the file path
            res[model_name] = extract_scores(abs_file_path)

        else:
            raise ValueError(f"Couldn't extract model name from {FILE_NAME}")
        
    # Now, convert the model map to a dictionary mapping file names to scores
    for model in res:
        results = res[model]  # List[dict]
        new_res = {}
        for result in results:  # {'rd_file': str, 'fd_file': str, 'scores': List[str]}
            # extract 243711d2cc from 'c:\\Users\\alexs\\Desktop\\judgement\\alma\\alma_docs\\alma_anonymized_final\\243711d2cc.txt'
            file_name = os.path.splitext(os.path.basename(result["fd_file"]))[0]
            new_res[file_name] = result["scores"]
        res[model] = new_res
    return res 


def map_raw_results_to_scores(forward_results_dir, reversed_results_dir):
    """
    Converts raw experiment data to a map between model names and scores.

    we have to read all files in. We have to create a map from file name to the scores. We have to create a map from model to scores.
    
    Returns:
    {
        "reversed": {  # the RD goes first, then the FD
            "model_name1": {
                "file1_name": scores1,  (x/5)
                "file2_name": scores2,
                ...
                },
            "model_name2": {
                "file1_name": scores3,
                "file2_name": scores4,
                ...
                }
            ...
        },
        "forward": {  # the FD goes first, then the RD
            "model_name1": {
                "file_name1": scores5,  (x/5)
                "file_name2": scores6,
            },
            "model_name2": {
                "file_name1": scores7,
                "file_name2": scores8,
            }
            ...
        }
    }
    """
    assert os.path.exists(forward_results_dir), f"Directory {forward_results_dir} not found."
    assert os.path.exists(reversed_results_dir), f"Directory {reversed_results_dir} not found."
    # format of the results: {model_name: {file_name: scores}}
    reversed_results, forward_results = map_files(reversed_results_dir), map_files(forward_results_dir)

    # convert all of the scores to numerical values
    for model in forward_results:
        for file in forward_results[model]:
            # print(f"Scores backward: {reversed_results[model][file]} --> {score_str_to_num(reversed_results[model][file], SCORE_CONVERSION_RD_FIRST)}")
            # print(f"Scores forward: {forward_results[model][file]} --> {score_str_to_num(forward_results[model][file], SCORE_CONVERSION_FD_FIRST)}")
            reversed_results[model][file] = score_str_to_num(reversed_results[model][file], SCORE_CONVERSION_RD_FIRST)
            forward_results[model][file] = score_str_to_num(forward_results[model][file], SCORE_CONVERSION_FD_FIRST)
    return {
        "reversed": reversed_results,
        "forward": forward_results
    }

"""
# We want both:
# - The average score for each model across all documents
# - The average score for each document across all models
To get model-wise scores, we have to add the sum of the scores across all files for both the forward and reversed directions
To get file 1's score, we have to add the sum of the scores across all models for both the forward and reversed directions
"""
def sum_directional_scores(results_map: dict):
    """
    Sums scores between the two directions.

    Returns a dictionary with the following structure:
    {
        "model_name1": {
            "file1": combined_score1,
            "file2": combined_score2,
            ...
        },
        "model_name2": {
            "file1": combined_score3,
            "file2": combined_score4,
            ...
        }, ...
    }
    """
    res = {}
    for model in results_map["forward"]:
        res[model] = {}
        for file in results_map["forward"][model]:
            res[model][file] = (results_map["forward"][model][file] + results_map["reversed"][model][file]) 
    return res


def retrieve_processed_scores(forward_results_dir: str, reversed_results_dir: str) -> Mapping[str, Mapping[str, float]]:
    """
    Retrieves the model scores from the raw results.

    Args:
        forward_results_dir (str): Directory containing the forward directional results (final draft first).
        reversed_results_dir (str): Directory containing the reversed directional results (rough draft first).

    Returns:
        dict: Mapping of model names to files to summed likert scale scores across all files.
        {
            "model_name1": {
                "file1": combined_score1,
                "file2": combined_score2,
                ...
            }, ...
        }
    """
    results_map = map_raw_results_to_scores(forward_results_dir, reversed_results_dir)
    return sum_directional_scores(results_map)


def get_per_model_score(processed_scores: Mapping[str, Mapping[str, float]]) -> Mapping[str, float]:
    """
    Gets the average score for each model across all documents.

    Args:
        processed_scores (Mapping[str, Mapping[str, float]]): Mapping of model names to files to summed likert scale scores across all files.
        (see `retrieve_processed_scores`)

    Returns:
        Mapping[str, float]: Dictionary mapping model names to average scores across all documents.
    """
    res = {}
    for model in processed_scores:
        scores = processed_scores[model].values()
        res[model] = sum(scores) / len(scores)
    return res


def get_per_file_score(processed_scores: Mapping[str, Mapping[str, float]]) -> Mapping[str, float]:
    """
    Gets the average score for each document across all models.

    Args:
        processed_scores (Mapping[str, Mapping[str, float]): Mapping of model names to files to summed likert scale scores across all files.
        (see `retrieve_processed_scores`)

    Returns:
        Mapping[str, float]: Dictionary mapping file names to average scores across all models.
    """
    res = {}
    for model in processed_scores:
        for file in processed_scores[model]:
            if file not in res:
                res[file] = 0
            res[file] += processed_scores[model][file]
    for file in res:
        res[file] /= len(processed_scores)
    return res



# Wrap everything in a main function that runs it backwards and forwards for some number of samples and averages everything
def retrieve_complete_scores(forward_results_dir: str, reversed_results_dir: str):
    """
    Retrieves the complete scores for the models.

    Args:
        forward_results_dir (str): Directory containing the forward directional results (final draft first).
        reversed_results_dir (str): Directory containing the reversed directional results (rough draft first).

    Returns:
        Mapping[str, float]: Dictionary mapping model names to average scores across all documents.
        Mapping[str, float]: Dictionary mapping file names to average scores across all models.
    """
    processed_scores = retrieve_processed_scores(forward_results_dir, reversed_results_dir)
    return get_per_model_score(processed_scores), get_per_file_score(processed_scores)


if __name__ == "__main__":
    ## Scoring conversions ##

    per_model, per_file = retrieve_complete_scores(
        forward_results_dir=r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison\demo\standard_order",
        reversed_results_dir=r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison\demo\reversed_order"
    )

    pprint.pprint(per_model)
    pprint.pprint(per_file)


