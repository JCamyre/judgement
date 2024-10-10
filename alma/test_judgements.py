import os
import re
import logging
from typing import List, Mapping, Any
import litellm 
from langfuse.decorators import observe
import pprint
# litellm.set_verbose = True  # for debugging

from judgement import * 
from judgement.constants import *
from judgement.prompt_names import *
from judgement.eval.model_graded import comparisons

### CAPTURING SCORES ###

def capture_verdict(text: str):
    # Regex pattern to capture one of the five patterns
    pattern = r"\[\[A(?:<<|<|=|>|>>)B\]\]"
    
    # Search for the pattern in the input text
    match = re.search(pattern, text)
    
    # Return the matched pattern if found
    if match:
        return match.group(0)
    else:
        return None


def test_single_judge(
        sampling_k: int = 1, 
        judge: str = LLAMA3_70B_INSTRUCT_TURBO, 
        input_rd_dir: str = None,
        input_fd_dir: str = None,
        output_dir: str = None
        ):
    """
    Prompts a single judge to compare the rough draft and final draft of a document based on criteria.
    The judge can be sampled independently `sampling_k` number of times to produce a distribution of scores.

    Args:
        sampling_k (int, optional): Number of times to sample from the judge. Defaults to 1.
        judge (str, optional): Judge model. Defaults to LLAMA3_70B_INSTRUCT_TURBO.
        output_dir (str, optional): Dir to write outputs to. Defaults to None.

    Outputs will be structured in files with the following format:
    {judge}_results.txt
    """
    llm_judge = comparisons.ComparisonEvaluator(
        judge=judge,
        eval_prompt_skeleton=langfuse.get_prompt(LETTER_COMPARISON),
    )

    rd_root = input_rd_dir
    fd_root = input_fd_dir
    out_file = os.path.join(output_dir, f"{judge}_results.txt")
    
    with open(out_file, "w") as out:
        for FILE_NAME in os.listdir(rd_root):
            rd_file = os.path.join(rd_root, FILE_NAME)
            fd_file = os.path.join(fd_root, FILE_NAME)
            out.write(f"Comparing {rd_file} and {fd_file} using judge {judge}")
            logging.info(f"Comparing {rd_file} and {fd_file} using judge {judge}. Writing to {out_file}")

            with open(rd_file, "r") as rough_draft_file, open(fd_file, "r") as final_draft_file:
                rd_content = rough_draft_file.read()
                fd_content = final_draft_file.read()
            # Sample 
            if sampling_k == 1:
                evaluation = llm_judge.evaluate_sample(criteria=BASE_LETTER_COMPARISON_CRITERIA, pred=rd_content, gold=fd_content)
                score = capture_verdict(evaluation)
                out.write(f"Score: {score}\n")
            else:  # sampling multiple times
                scores = []
                evaluation = llm_judge.evaluate_samples_batch(criteria=BASE_LETTER_COMPARISON_CRITERIA, 
                                                            preds=[rd_content] * sampling_k, 
                                                            golds=[fd_content] * sampling_k)  # List of responses from the judge model
                for e in evaluation:
                    score = capture_verdict(e)
                    if score is None:  # couldn't detect answer
                        logging.warning(f"Couldn't detect answer from {e}")
                    scores.append(score)
                out.write(f"Scores: {scores}\n")
                logging.info(f"Scores: {scores}")
            

@observe
def test_mixture_of_judges(
    sampling_k: int,
    judges: list[str],
    aggregator: str,
    criteria: str,
    input_rd_dir: str = None,
    input_fd_dir: str = None,
    output_path: str = None
    ):
    judge_mixture = comparisons.ComparisonMixture(
        judges=judges,
        aggregator=aggregator,
        eval_prompt_skeleton=langfuse.get_prompt(LETTER_COMPARISON),
        mixture_base_prompt=langfuse.get_prompt(MIXTURE_OF_JUDGES),
    )
    for FILE_NAME in os.listdir(input_rd_dir)[:3]:
        logging.info(f"Comparing {FILE_NAME} using judges {judges} and aggregator {aggregator}")
        print(f"Comparing {FILE_NAME} using judges {judges} and aggregator {aggregator}")
        rd_file = os.path.join(input_rd_dir, FILE_NAME)
        fd_file = os.path.join(input_fd_dir, FILE_NAME)

        with open(rd_file, "r") as rough_draft_file, open(fd_file, "r") as final_draft_file:
            rd_content = rough_draft_file.read()
            fd_content = final_draft_file.read()
        scores = []
        for i in range(sampling_k):
            evaluation = judge_mixture.evaluate_sample(
                criteria=criteria, 
                pred=rd_content, 
                gold=fd_content
                )
            score = capture_verdict(evaluation)
            scores.append(score)
            if score is None:
                logging.warning(f"Couldn't detect answer from {evaluation}")
                print(f"Couldn't detect answer from {evaluation}")
        with open(output_path, "a") as out:
            out.write(f"Comparing {rd_file} and {fd_file} using judges {judges} and aggregator {aggregator}\n")
            out.write(f"Scores: {scores}\n")
            logging.info(f"Comparing {rd_file} and {fd_file} using judges {judges} and aggregator {aggregator}. Scores: {scores}")
            print(f"Comparing {rd_file} and {fd_file} using judges {judges} and aggregator {aggregator}. Scores: {scores}")


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
    pattern = re.compile(r"Comparing (.+?) and (.+?) using judge .+? Scores: (.+)")

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
                "file1_name": scores1,
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
                "file_name1": scores5,
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
    # # TODO: You may need to update your paths
    # OUTPUT_DIR = r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison"
    # CRITERIA_PATH = r"C:\Users\alexs\Desktop\judgement\alma\criteria\machine_generated_v1.txt"
    # with open(CRITERIA_PATH, "r") as f:
    #     letter_criteria = f.read()
    #     # Running tests with a single judge
    #     # JUDGES = [QWEN, LLAMA3_70B_INSTRUCT_TURBO, LLAMA3_405B_INSTRUCT_TURBO, 
    #     #         LLAMA3_8B_INSTRUCT_TURBO, MISTRAL_8x22B_INSTRUCT, MISTRAL_8x7B_INSTRUCT]
    #     # # rd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft")
    #     # # fd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final")
    #     # for judge in JUDGES:
    #     #     test_single_judge(sampling_k=3, judge=judge, output_dir=OUTPUT_DIR)

    #     # test_single_judge(sampling_k=3, judge=LLAMA3_70B_INSTRUCT_TURBO, output_dir=OUTPUT_DIR)

    #     # Running tests with mixture of models
    #     test_mixture_of_judges(
    #         sampling_k=3,
    #         judges=[QWEN, 
    #                 LLAMA3_70B_INSTRUCT_TURBO, 
    #                 MISTRAL_8x22B_INSTRUCT],
    #         aggregator=QWEN,
    #         criteria=letter_criteria,
    #         input_rd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft"),
    #         input_fd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final"),
    #         output_path=os.path.join(OUTPUT_DIR, "mixture_results.txt")
    #     )


    ## Scoring conversions ##

    per_model, per_file = retrieve_complete_scores(
        forward_results_dir=r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison\standard_order",
        reversed_results_dir=r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison\reversed_order"
    )

    pprint.pprint(per_model)
    pprint.pprint(per_file)

