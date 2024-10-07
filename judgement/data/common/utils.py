"""
This file contains utility functions used in data cleaning scripts
"""

from judgement import *
from judgement.constants import *
from typing import List, Mapping, Dict, Union
import litellm
import pydantic
import pprint
import os
from dotenv import load_dotenv 

load_dotenv()

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def get_chat_completion(model_type: str, 
                        messages : Union[List[Mapping], List[List[Mapping]]], 
                        response_format: pydantic.BaseModel = None, 
                        batched: bool = False
                        ) -> Union[str, List[str]]:
    """
    Generates chat completions using specified model and messages.

    Parameters:
        - model_type (str): The type of model to use for generating completions.
        - messages (Union[List[Mapping], List[List[Mapping]]]): The messages to be used for generating completions. 
            If batched is True, this should be a list of lists of mappings.
        - response_format (pydantic.BaseModel, optional): The format of the response. Defaults to None.
        - batched (bool, optional): Whether to process messages in batch mode. Defaults to False.
    Returns:
        - str: The generated chat completion(s). If batched is True, returns a list of strings.
    Raises:
        - ValueError: If messages are not in the correct format for single completion.
    """
    if batched:
        if type(messages[0]) != list:
            raise ValueError("Messages must be a list of lists of dictionaries for batch completion.")
        if model_type not in TOGETHER_SUPPORTED_MODELS:  # supported by Litellm
            if response_format:
                response = litellm.batch_completion(
                    model=model_type,
                    messages=messages,
                    response_format=response_format,
                )
            else:
                response = litellm.batch_completion(
                    model=model_type,
                    messages=messages,
                )
            out = [r.choices[0].message.content for r in response] 
        else:  # using Together client instead.  TODO figure out how batch inference works for together
            out = []
            for m in messages:
                response = together_client.chat.completions.create(
                    model=TOGETHER_SUPPORTED_MODELS.get(model_type),
                    messages=m,
                )
                out.append(response.choices[0].message.content)
        return out
    else:  # single completion
        if type(messages[0]) == list:
            raise ValueError("Messages must be a list of dictionaries for single completion.") 

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


def get_completion_multiple_models(models: List[str], message: List[Mapping], response_format: pydantic.BaseModel = None) -> List[str]:
    """
    Retrieves completions to a prompt from multiple models (in parallel)
    """

    # for now, we will use the Litellm client to get completions from multiple models
    # if we want to call the together models, we gotta wait
    for m in models:
        if m in TOGETHER_SUPPORTED_MODELS:
            raise ValueError(f"Model {m} is not supported by Litellm for multiple completions.")
    all_responses = litellm.batch_completion_models_all_responses(
                models=models, 
                messages=message,    
                response_format=response_format
                )
    return [response.choices[0].message.content for response in all_responses] 


if __name__ == "__main__":
    
    # Batched
    pprint.pprint(get_chat_completion(
        model_type=GPT4_MINI,
        messages=[
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ]
        ],
        batched=True
    ))

    # Non batched
    pprint.pprint(get_chat_completion(
        model_type=GPT4_MINI,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        batched=False
    ))

    # Batched single completion to multiple models
    pprint.pprint(get_completion_multiple_models(
        models=[GPT4_MINI, CLAUDE_SONNET],
        message=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    ))
