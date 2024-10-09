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
    # TODO: Improve this prompt
    combined_input = f"Context: {context}\nQuestion: {question}"

    example = dspy.Example(question=combined_input)
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

# TODO: Add call to Arena-Hard-esque LLM as a judge
def custom_metric(output, reference, trace=None) -> float:
    print(f"{output=}, {reference=}, {trace=}")

teleprompter = BootstrapFewShot(metric=custom_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=4)

optimized_cot = teleprompter.compile(CoT(), trainset=alma_trainset)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=alma_devset, metric=custom_metric, num_threads=4, display_progress=True, display_table=0)

# # Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

print(turbo.inspect_history(n=1))
