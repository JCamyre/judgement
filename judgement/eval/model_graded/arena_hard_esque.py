from judgement.eval.model_graded.judge import LLMJudge
from judgement.data.common import utils

# 1. Make Langfuse prompt + add prompt to constants
    # Will have to add the variables in the Langfuse prompt for the two criteria
# 2. Basically copy comparisons.py ComparisonEvaluator, do the mixture of judges later
# 3. 

class ArenaHardJudge(LLMJudge):
    
    def evaluate_sample(self, rough_draft: str, final_draft: str, baseline_criteria: str, candidate_criteria: str):
        compiled_eval_prompt = self.eval_prompt_skeleton.compile(
            rough_draft=rough_draft,
            final_draft=final_draft,
            first_criteria=baseline_criteria,
            second_criteria=candidate_criteria
        )
        
        chat_completion = utils.get_chat_completion(self.judge, compiled_eval_prompt)
        return chat_completion
    
    def evaluate_samples_batch(self, *args, **kwargs) -> utils.List[str]:
        return super().evaluate_samples_batch(*args, **kwargs)
    
    def evaluate_test_set(self, *args, **kwargs) -> utils.List[str]:
        return super().evaluate_test_set(*args, **kwargs)