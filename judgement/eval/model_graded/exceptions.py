"""
Exceptions for the model_graded module.
"""

class JudgeNotSupportedError(Exception):
    def __init__(self, judge: str):
        self.judge = judge
        super().__init__(f"Model {judge} is not supported by Litellm or Together for completions.")