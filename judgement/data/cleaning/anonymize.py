"""
Takes a document and anonymizes it by replacing all names with alternative names.
"""

from judgement import langfuse, client
from judgement.data.cleaning import utils
from judgement.prompt_names import *
from judgement.constants import *
import os
import pprint


def replace_identifying_info(document_path: str, model_type: str = "gpt-4o-mini") -> str:
    """
    Anonymizes a document by replacing all identifying information such as names, places, companies, etc., with alternative, non-identifying information.
    Replace names with other common names, places with other cities or regions, and companies with other generic or fictional company names.

    Args:
        document_path (str): The path to the document to be anonymized.
        model_type (str): The type of model to use for anonymization. Defaults to "gpt-4o-mini".
    Returns:
        str: The anonymized document.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"File not found at path: {document_path}")
    # Load in prompt for this task
    prompt = langfuse.get_prompt(   # fetches the prompt from the langfuse client
        ANONYMIZATION_PROMPT,  
        type="chat"
    )
    compiled_prompt = prompt.compile(
        mask_doc=utils.read_file(document_path)   # mask_doc is the langfuse prompt param for the document to be anonymized
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= compiled_prompt, 
    )
    # Extract the anonymized document from the response
    anonymized_document = completion.choices[0].message.content
    return anonymized_document

if __name__ == "__main__":
    document = os.path.join(os.path.dirname(__file__), "samples", "example_letter.txt")

    anonymized_document = replace_identifying_info(document)
    print("*" * 50)
    print(anonymized_document)
