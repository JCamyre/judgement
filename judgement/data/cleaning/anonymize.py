"""
Takes a document and anonymizes it by replacing all names with alternative names.
"""

from judgement import *
from judgement.data.cleaning import utils
from judgement.prompt_names import *
from judgement.constants import *
import os
import pprint
import litellm
import pydantic
from typing import Dict

class DocumentOutput(pydantic.BaseModel):
    draft_doc: str
    final_doc: str

def replace_identifying_info_doc_pairs(draft_document_path: str, final_document_path: str, response_format: pydantic.BaseModel, model_type: str = GPT4_MINI) -> Dict[str, str]:
    if not os.path.exists(draft_document_path):
        raise FileNotFoundError(f"File not found at path: {draft_document_path}")
    
    if not os.path.exists(final_document_path):
        raise FileNotFoundError(f"File not found at path: {final_document_path}")
    
    prompt = langfuse.get_prompt(   # fetches the prompt from the langfuse client
        ANONYMIZATION_PAIR_PROMPT,  
        type="chat"
    )
        
    # Add logic to concentate documents
    with open(draft_document_path, "r") as file:
        draft_content = file.read()
        
    with open(final_document_path, "r") as file:
        final_content = file.read()
        
    concatenated_content = f"First email: {draft_content}\nSecond email: {final_content}"
    
    compiled_prompt = prompt.compile(
        mask_doc=concatenated_content   # mask_doc is the langfuse prompt param for the document to be anonymized
    )
    
    chat_completion = utils.get_chat_completion(model_type, compiled_prompt, response_format=response_format)
    return response_format.model_validate_json(chat_completion).model_dump()

def replace_identifying_info(document_path: str, model_type: str = GPT4_MINI) -> str:
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
    # Extract the anonymized document from the response
    return utils.get_chat_completion(model_type, compiled_prompt)

if __name__ == "__main__":
    # document = os.path.join(os.path.dirname(__file__), "samples", "example_letter.txt")
    # anonymized_document = replace_identifying_info(document, model_type=GPT4_MINI)
    # print("*" * 50)
    # print(anonymized_document)
    draft_document_path = os.path.join(os.path.dirname(__file__), "samples", "example_draft.txt")
    final_document_path = os.path.join(os.path.dirname(__file__), "samples", "example_final.txt")
    anonymized_documents = replace_identifying_info_doc_pairs(draft_document_path, final_document_path, DocumentOutput)
    print("*" * 50)
    with open(os.path.join(os.path.dirname(__file__), "samples", "example_draft_anonymized.txt"), "w") as file:
        file.write(anonymized_documents["draft_doc"])
        
    with open(os.path.join(os.path.dirname(__file__), "samples", "example_final_anonymized.txt"), "w") as file:
        file.write(anonymized_documents["final_doc"])
