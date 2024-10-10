import os
import re
import logging
from typing import List, Mapping, Any
from langfuse.decorators import observe
# litellm.set_verbose = True  # for debugging

from judgement import * 
from judgement.constants import *
from judgement.prompt_names import *
from judgement.eval.model_graded import comparisons

### CAPTURING SCORES ###

def capture_verdict(text: str) -> str:
    """
    Captures the score from the text using a regex pattern. 

    Finds one of the five patterns:
    - [[A<<B]]
    - [[A<B]]
    - [[A=B]]
    - [[A>B]]
    - [[A>>B]]
    """
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
        sampling_k: int = 3, 
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
    )  # eval prompt skeleton is order dependent (you have to specify the order of the letters)

    rd_root = input_rd_dir
    fd_root = input_fd_dir
    out_file = os.path.join(output_dir, f"{judge}_results.txt")
    
    with open(out_file, "w") as out:
        for FILE_NAME in os.listdir(rd_root):
            rd_file = os.path.join(rd_root, FILE_NAME)
            fd_file = os.path.join(fd_root, FILE_NAME)
            out.write(f"Comparing {rd_file} and {fd_file} using judge {judge} ")
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
                    score = capture_verdict(e)   # capture score, e.g. '[[A>>B]]'
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
    for FILE_NAME in os.listdir(input_rd_dir):
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
            out.write(f"Comparing {rd_file} and {fd_file} using judges {"/".join(judges) + f"+{aggregator}"} Scores: {scores}\n")
            logging.info(f"Comparing {rd_file} and {fd_file} using judges {judges} and aggregator {aggregator}. Scores: {scores}")
            print(f"Comparing {rd_file} and {fd_file} using judges {judges} and aggregator {aggregator}. Scores: {scores}")


if __name__ == "__main__":
    # TODO: You may need to update your paths
    OUTPUT_DIR = r"C:\Users\alexs\Desktop\judgement\alma\alma_results\letter_comparison"
    CRITERIA_PATH = r"C:\Users\alexs\Desktop\judgement\alma\criteria\machine_generated_v1.txt"

    with open(CRITERIA_PATH, "r") as f:
        letter_criteria = f.read()
        # Running tests with a single judge
        JUDGES = [QWEN, LLAMA3_70B_INSTRUCT_TURBO, LLAMA3_8B_INSTRUCT_TURBO, MISTRAL_8x22B_INSTRUCT, MISTRAL_8x7B_INSTRUCT]
        for judge in JUDGES:
            test_single_judge(
                sampling_k=3, 
                judge=judge, 
                input_rd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft"),
                input_fd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final"),
                output_dir=OUTPUT_DIR
            )

        # Running tests with mixture of models
        test_mixture_of_judges(
            sampling_k=3,
            judges=[QWEN, 
                    LLAMA3_70B_INSTRUCT_TURBO, 
                    MISTRAL_8x22B_INSTRUCT],
            aggregator=QWEN,
            criteria=letter_criteria,
            input_rd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft"),
            input_fd_dir=os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final"),
            output_path=os.path.join(OUTPUT_DIR, "QWEN_L3_70B_MISTRAL8x22B-QWEN_results.txt")
        )

