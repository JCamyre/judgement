import dspy
from dspy.datasets.gsm8k import GSM8K
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
import pandas as pd

turbo = dspy.OpenAI(model='gpt-4o-mini', max_tokens=250)
dspy.settings.configure(lm=turbo)

df = pd.read_csv("./judgement/eval/output.csv")

dataset = []

for context, question, answer in df.values:
    combined_input = f"Context: {context}\nQuestion: {question}"

    example = dspy.Example(question=combined_input, answer=answer)
    example = example.with_inputs("question")
    dataset.append(example)
    
alma_trainset, alma_devset = dataset[:7], dataset[7:]
# print(f"{type(alma_trainset[0])=}")

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
    
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# TODO: Add call to Arena Hard Auto
def custom_metric(output, reference, trace=None) -> float:
    print(f"{output=}, {reference=}, {trace=}")
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

# # teleprompter = BootstrapFewShotWithRandomSearch(metric=custom_metric, **config)
teleprompter = BootstrapFewShotWithRandomSearch(metric=custom_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=4)

optimized_cot = teleprompter.compile(CoT(), trainset=alma_trainset)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=alma_devset, metric=custom_metric, num_threads=4, display_progress=True, display_table=0)

# # Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

print(turbo.inspect_history(n=1))
