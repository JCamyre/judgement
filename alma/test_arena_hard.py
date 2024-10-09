import os
import random

from judgement.eval.model_graded import arena_hard_esque
from judgement.constants import GPT4O
from judgement.prompt_names import ARENA_HARD_ESQUE

def test_single_judge():
    llm_judge = arena_hard_esque.ArenaHardJudge(
        judge=GPT4O,
        eval_prompt_skeleton=ARENA_HARD_ESQUE
    )
    
    # Load in baseline criteria
    baseline_criteria = ""
    
    # Pretend this is from a custom_metrics call: output: str, assume it's just a single criteria
    dspy_criteria = ""
    
    # Get some random rough and final draft pair
    rd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft")
    fd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final")
    
    file_list = [f for f in os.listdir(rd_root) if os.path.isfile(os.path.join(rd_root, f))]
    rough_draft = ""
    final_draft = ""
    if file_list:
        # Select a random file from the list
        random_file = random.choice(file_list)
        print(f"Random file selected: {random_file}")
        with open(os.path.join(rd_root, rough_draft), "r") as file:
            rough_draft = file.read()
        
        final_file_path = os.path.join(fd_root, random_file)

        # Check if the corresponding file exists in the final directory
        if os.path.exists(final_file_path):
            with open(final_file_path, "r") as file:
                final_draft = file.read()
    else:
        print("The directory is empty.")

    evaluation = llm_judge.evaluate_sample(rough_draft, final_draft, baseline_criteria, dspy_criteria)
    print(evaluation)
    
