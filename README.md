# judgement

## Environment Set-Up

1. Clone repo
2. In root directory, run `pipenv shell`
   1. If you don't have `pipenv` installed, install with `pip install pipenv`
3. Run `pipenv install`, which will install the packages and use the Python version specified in the Pipfile
4. Create `.env` file in root directory, adding all secret keys and setting the Python path to the root directory by adding `PYTHONPATH="."`

Whenever you open a new terminal, access your pipenv environment via `pipenv shell`

## Evaluating using LLM as a Judge
The `LLMJudge` base class in `judgement/eval/model_graded/judge.py` implements an abstract class of an LLM as a Judge. 
To create an `LLMJudge`, you must provide

1. `judge (str)`: A foundation model to execute the judgement with. Used names can be found in `judgement/constants.py`.
2. `eval_prompt_skeleton (langfuse.client.ChatPromptClient)`: A loaded prompt object from langfuse which sets up the specific evaluation task for the judge.

To write an instance of an `LLMJudge`, you must write the following abstract methods:

1. `evaluate_sample`: Takes in a specified input (ex: a pair of predicted and gold outputs for a task) and invokes the judge to produce an evaluation over the inputs.
2. `evaluate_samples_batch`: A batched version of `evaluate_sample`
3. `evaluate_test_set`: A version of `evaluate_sample` that executes evaluation across an entire dataset (the `Dataset` object can be found at `judgement/data/common/dataset.py`).

An example implementation of the `LLMJudge` abstract class can be found in `judgement/eval/model_graded/comparisons.py`.

## Evaluating using Mixture of Judges
The `MixtureofJudges` base class in `judgement/eval/model_graded/judge.py` implements an abstract class of Mixture of Judges.
To create a `MixtureofJudges`, you must provide

1. `judges (List[str])`: A list of foundation model names to use as the judges.
2. `aggregator (str)`: Model name for the aggregator LLM that puts judge responses together.
3. `eval_prompt_skeleton (langfuse.client.ChatPromptClient)`: A loaded prompt object from langfuse which sets up the specific evaluation task for each judge.
4. `mixture_base_prompt (lf.client.ChatPromptClient)`: A loaded langfuse prompt object which gives the context to merge prompts together via the aggregator.

To write an instance of `MixtureofJudges`, you must write the following abstract methods:

1. `build_dynamic_mixture_prompt`: Builds a dynamic prompt for mixing the judge responses, depending on the number of judges.
2. `evaluate_sample`: Takes in a specified input (ex: a pair of predicted and gold outputs for a task) and invokes the judge to produce an evaluation over the inputs.
3. `evaluate_samples_batch`: A batched version of `evaluate_sample`
4. `evaluate_test_set`: A version of `evaluate_sample` that executes evaluation across an entire dataset (the `Dataset` object can be found at `judgement/data/common/dataset.py`).

An example implementation of the `MixtureofJudges` abstract class can be found in `judgement/eval/model_graded/comparisons.py`.
