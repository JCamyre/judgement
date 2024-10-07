import os
from judgement import * 
from judgement.constants import *
from judgement.prompt_names import *
from judgement.eval.model_graded import judge 



def test_single_judge():
    llm_judge = judge.LLMJudge(
        judge=GPT4O,
        eval_prompt_skeleton=langfuse.get_prompt(LETTER_COMPARISON),
    )

    rd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft")
    fd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final")

    FILE_NAME = "44866f97ba.txt"
    rd_file = os.path.join(rd_root, FILE_NAME)
    fd_file = os.path.join(fd_root, FILE_NAME)

    with open(rd_file, "r") as rough_draft_file, open(fd_file, "r") as final_draft_file:
        rd_content = rough_draft_file.read()
        fd_content = final_draft_file.read()

    evaluation = llm_judge.evaluate_sample(criteria=BASE_LETTER_COMPARISON_CRITERIA, pred=rd_content, gold=fd_content)
    print(evaluation)


def test_mixture_of_judges():
    rd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_draft")
    fd_root = os.path.join(os.path.dirname(__file__), "alma_docs", "alma_anonymized_final")

    FILE_NAME = "44866f97ba.txt"
    rd_file = os.path.join(rd_root, FILE_NAME)
    fd_file = os.path.join(fd_root, FILE_NAME)

    with open(rd_file, "r") as rough_draft_file, open(fd_file, "r") as final_draft_file:
        rd_content = rough_draft_file.read()
        fd_content = final_draft_file.read()
    
    judge_mixture = judge.MixtureofJudges(
        judges=[GPT4O, CLAUDE_SONNET, GPT4_MINI],
        aggregator=GPT4O,
        eval_prompt_skeleton=langfuse.get_prompt(LETTER_COMPARISON),
        mixture_base_prompt=langfuse.get_prompt(MIXTURE_OF_JUDGES),
    )
    evaluation = judge_mixture.evaluate_sample(criteria=BASE_LETTER_COMPARISON_CRITERIA, pred=rd_content, gold=fd_content)
    print(evaluation)


if __name__ == "__main__":
    test_single_judge()
    test_mixture_of_judges()


