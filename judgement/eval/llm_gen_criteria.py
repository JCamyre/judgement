import os
import pydantic

from judgement import langfuse
from judgement.constants import GPT4_MINI
from judgement.prompt_names import LLM_GEN_CRITIERIA
from judgement.data.cleaning.utils import read_file, get_chat_completion

class CriteriaOutput(pydantic.BaseModel):
    CRITERIA: str
    THOUGHTS: str

# TODO: Append the criteria to the next LLM call. Have to update the prompt to inform LLM about it.
def generate_criteria(rough_draft_path: str, final_draft_path: str, model_type: str = GPT4_MINI, response_format: str = CriteriaOutput):
    prompt = langfuse.get_prompt(
        LLM_GEN_CRITIERIA,
        type="chat"
    )
    
    # Load document pair
    rough_draft = read_file(rough_draft_path)
        
    final_draft = read_file(final_draft_path)
    
    criteria = None
        
    raw_prompt = f"Rough draft: {rough_draft}\nFinal draft: {final_draft}\Current criteria: {criteria}"
    
    compiled_prompt = prompt.compile(
        criteria=raw_prompt
    )
    
    # Format the response into two parts: the explanation for everything, and the criteria
    
    chat_completion = get_chat_completion(model_type, compiled_prompt, response_format=response_format)
    return response_format.model_validate_json(chat_completion).model_dump()

if __name__ == "__main__":
    # Testing
    rough_draft_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_draft", "243711d2cc.txt")
    final_draft_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_final", "243711d2cc.txt")
    output = generate_criteria(rough_draft_path, final_draft_path)
    criteria = output["CRITERIA"]
    thoughts = output["THOUGHTS"]
    print(f"{criteria=}, {thoughts=}")
    print("*" * 50)
    
