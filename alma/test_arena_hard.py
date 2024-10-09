import os
import random
from time import sleep
import pydantic

from judgement import langfuse
from judgement.eval.model_graded import arena_hard_esque
from judgement.constants import GPT4_MINI, GPT4O, BASE_LETTER_COMPARISON_CRITERIA
from judgement.prompt_names import ARENA_HARD_ESQUE

class JudgeOutput(pydantic.BaseModel):
    REASONING: str
    SCORE: str
    
# TODO: For later, test mixture of judges.
# Test the order of baseline and DSPy criteria
# Pretend this is from a custom_metrics call: output: str, assume it's just a single criteria
def test_single_judge(dspy_criteria: str):
    llm_judge = arena_hard_esque.ArenaHardJudge(
        judge=GPT4_MINI,
        eval_prompt_skeleton=langfuse.get_prompt(ARENA_HARD_ESQUE)
    )
    
    # Load in baseline criteria from previous LLM-generated criteria.
    baseline_criteria = BASE_LETTER_COMPARISON_CRITERIA
    
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
        with open(os.path.join(rd_root, random_file), "r") as file:
            rough_draft = file.read()
        
        final_file_path = os.path.join(fd_root, random_file)

        # Check if the corresponding file exists in the final directory
        if os.path.exists(final_file_path):
            with open(final_file_path, "r") as file:
                final_draft = file.read()
    else:
        print("The directory is empty.")

    evaluation = llm_judge.evaluate_sample(rough_draft, final_draft, baseline_criteria, dspy_criteria, JudgeOutput)
    # Get the score from this, return it. Do something with the reasoning later.
    output = JudgeOutput.model_validate_json(evaluation).model_dump()
    score, reasoning = output["SCORE"], output["REASONING"]
    
    # TODO Find better way to get around GPT rate limit
    # sleep(45)
    # Return a 0-1 float score.
    return score

if __name__ == "__main__":    
    TEST_LETTER_COMPARISON_CRITERIA = """1. Conciseness: Evaluate how effectively the letter conveys important information without unnecessary details or wordiness.
    2. Fluidity: Assess the smoothness of the language and sentence transitions, ensuring the letter reads naturally and flows well.
    3. Clarity: Consider how clearly the information is communicated. Are the ideas easy to follow, with no ambiguity or confusion?
    4. Tone and Formality: Judge whether the tone is professional and appropriate for an immigration letter, reflecting the seriousness of the matter."""
    print(test_single_judge(TEST_LETTER_COMPARISON_CRITERIA))

