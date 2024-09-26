"""
This file contains utility functions used in data cleaning scripts
"""
from judgement import *
from judgement.constants import *
from typing import List, Mapping, Dict
import litellm
import pydantic

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def get_chat_completion(model_type: str, messages : List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    Get the completion of a chat conversation using the specified model type.
    Model types are listed in the constants file.
    Args:
        model_type (str): The type of model to use for completion.
        messages (List[Mapping]): The list of messages in the chat conversation.
    Returns:
        str: The completion of the chat conversation.
    Raises:
        None
    """
    if model_type not in TOGETHER_SUPPORTED_MODELS:  # supported by Litellm
        if response_format:
            response = litellm.completion(
                model=model_type,
                messages=messages,
                response_format=response_format
            )
        else:
            response = litellm.completion(
                model=model_type,
                messages=messages,
            )
    else:  # using Together client instead.
        response = together_client.chat.completions.create(
            model=TOGETHER_SUPPORTED_MODELS.get(model_type),
            messages=messages,
        )
    out = response.choices[0].message.content
    return out
