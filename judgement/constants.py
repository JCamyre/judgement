"""
Constants for the Judgement module.
"""

# Evaluation params
MAX_WORKER_THREADS = 10

# Model names
GPT4_MINI = "gpt-4o-mini"
GPT4O = "gpt-4o"
CLAUDE_SONNET = "claude-3-5-sonnet-20240620"
QWEN = "QWEN"
LLAMA3_70B_INSTRUCT_TURBO = "LLAMA3_70B_INSTRUCT_TURBO"
TOGETHER_SUPPORTED_MODELS = {
    "QWEN": "Qwen/Qwen2-72B-Instruct",
    "LLAMA3_70B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
}

BASE_LETTER_COMPARISON_CRITERIA = """1. Conciseness: Evaluate how effectively the letter conveys important information without unnecessary details or wordiness.
2. Fluidity: Assess the smoothness of the language and sentence transitions, ensuring the letter reads naturally and flows well.
3. Clarity: Consider how clearly the information is communicated. Are the ideas easy to follow, with no ambiguity or confusion?
4. Tone and Formality: Judge whether the tone is professional and appropriate for an immigration letter, reflecting the seriousness of the matter.
5. Logical Consistency: Check if the arguments and points are well-structured and logically sound, with no contradictions or gaps.
6. Persuasiveness: Determine how compelling and convincing the letter is in presenting its case.
7. Attention to Detail: Examine if the letter avoids errors, omissions, or vague statements that could weaken its argument."""
