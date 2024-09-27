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


if __name__ == "__main__":
    # They'll need to have the same name
    alma_documents_path = os.path.join(os.path.dirname(__file__), "alma_draft_documents")
    alma_final_path = os.path.join(os.path.dirname(__file__), "alma_final_documents")

    # Loop through all files in alma_draft_documents
    for draft_document_name in os.listdir(alma_documents_path):
        # Construct the path for the file in alma_draft_documents
        draft_document_path = os.path.join(alma_documents_path, draft_document_name)

        # Construct the corresponding path in alma_final_documents
        final_document_path = os.path.join(alma_final_path, draft_document_name)

        # Check if the file exists in alma_final_documents
        if os.path.exists(final_document_path):
            print(f"Found corresponding file: {final_document_path}")
            anonymized_documents = replace_identifying_info_doc_pairs(draft_document_path, final_document_path, DocumentOutput)
            with open(os.path.join(os.path.dirname(__file__), "alma_anonymized_draft", draft_document_name), "w") as file:
                file.write(anonymized_documents["draft_doc"])
                
            with open(os.path.join(os.path.dirname(__file__), "alma_anonymized_final", draft_document_name), "w") as file:
                file.write(anonymized_documents["final_doc"])
        else:
            print(f"No corresponding file for: {draft_document_name}")
