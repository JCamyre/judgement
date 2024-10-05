import os
import pydantic

from judgement import langfuse
from judgement.constants import GPT4_MINI
from judgement.prompt_names import LLM_GEN_CRITIERIA
from judgement.data.cleaning.utils import read_file, get_chat_completion

class CriteriaOutput(pydantic.BaseModel):
    CRITERIA: str
    THOUGHTS: str

# Have it give examples for the criteria, once we aggregate all the criteria, it can have hard examples of 1, 3, 5 for each criteria
# TODO: Learn more about what LLM as a judge likes for criteria
# Alignment can come later
# How can I use the "thoughts" to help
# Apply more prompt engineering techniques
# "such that we can help immigrants enter the US"
# TODO Small write up on my prompting

def generate_criteria(rough_draft_path: str, final_draft_path: str, criteria: str = "", model_type: str = GPT4_MINI, response_format: str = CriteriaOutput):
    prompt = langfuse.get_prompt(
        LLM_GEN_CRITIERIA,
        type="chat"
    )
    
    # Load document pair
    rough_draft = read_file(rough_draft_path)
        
    final_draft = read_file(final_draft_path)
        
    raw_prompt = f"Rough draft: {rough_draft}\nFinal draft: {final_draft}\Current criteria: {criteria}"
    
    compiled_prompt = prompt.compile(
        criteria=raw_prompt
    )
    
    # Format the response into two parts: the explanation for everything, and the criteria
    
    chat_completion = get_chat_completion(model_type, compiled_prompt, response_format=response_format)
    return response_format.model_validate_json(chat_completion).model_dump()

if __name__ == "__main__":
    with open("./test.txt", "w") as file:
        file.write("")
    
    alma_documents_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_draft")
    alma_final_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_final")
    criteria = ""
    for draft_document_name in os.listdir(alma_documents_path):
        rough_draft_path = os.path.join(alma_documents_path, draft_document_name)

        final_draft_path = os.path.join(alma_final_path, draft_document_name)

        if os.path.exists(final_draft_path):
            output = generate_criteria(rough_draft_path, final_draft_path, criteria)
            criteria = output["CRITERIA"]
            thoughts = output["THOUGHTS"]
            with open("./test.txt", "+a") as file:
                file.write(f"Criteria:\n{criteria}\n\nThoughts:\n{thoughts}\n")
                file.write("*" * 50 + "\n")
            print(f"{criteria=}, {thoughts=}")
            print("*" * 50)
        else:
            print(f"No corresponding file for: {draft_document_name}")
    
