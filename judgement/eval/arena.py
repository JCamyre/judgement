import dspy
from dspy.datasets.gsm8k import GSM8K
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
import litellm

from judgement.data.cleaning import utils
from judgement.constants import GPT4_MINI, CLAUDE_SONNET

# def litellm_completion(prompt, **kwargs):
#     prompt = {"content": prompt, "role": "user"}
#     return utils.get_chat_completion(model_type=CLAUDE_SONNET, messages=prompt)

# dspy.settings.configure(lm=litellm_completion)

turbo = dspy.OpenAI(model='gpt-3.5-turbo-0125', max_tokens=250)
dspy.settings.configure(lm=turbo)
print(f"{turbo.api_key=}")

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
# print(f"{gsm8k_trainset=}")

# TODO Step 1: Set up auto arena
def dummy_auto_arena():
    
    dummy_auto_arena_output = "gpt-3.5-turbo-0125             | score: 23.3  | 95% CI: (-2.1, 1.4)  | average #tokens: 329"
    return dummy_auto_arena_output

# Step 2: Use DSPY to optimize given the response from arena judge
# Task: Optimize the criteria prompt given to a LLM as a judge, which will score immigration letters.
# Metrics to maximize: The Arena Hard Auto score
# Few example inputs: Use the rough draft + final draft + generated criteria
# Each layer a signature: Layer 1: Input rough draft + final draft + prompt -> LLM-generated criteria.
# Layer 2: Auto arena output
# Optimizer: Compile this two step pipeline into "high-quality instructions"

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
    
# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# TODO: Add call to Arena Hard
def custom_metric(output, reference, trace=None) -> float:
    # print(f"{output=}, {reference=}, {trace=}")
    score = 23.3  # Replace with actual scoring logic
    confidence_interval = (-2.1, 1.4)  # Replace with actual CI calculation
    
    # Calculate the average of the confidence interval bounds
    ci_adjustment = (confidence_interval[0] + confidence_interval[1]) / 2
    
    # Adjust the score based on the confidence interval
    adjusted_score = score + ci_adjustment

    # Ensure the score is within a valid range [0, 100]
    adjusted_score = max(0, min(adjusted_score, 100))
    
    # Return the adjusted score as a percentage
    return adjusted_score / 100

# teleprompter = BootstrapFewShotWithRandomSearch(metric=custom_metric, **config)
teleprompter = BootstrapFewShotWithRandomSearch(metric=custom_metric, max_bootstrapped_demos=8, num_candidate_programs=8, num_threads=4)

optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=custom_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

# print(turbo.inspect_history(n=1))
