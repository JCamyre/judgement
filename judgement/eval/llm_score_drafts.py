import os
import csv
import pydantic

from judgement import langfuse
from judgement.constants import GPT4_MINI, BASE_LETTER_COMPARISON_CRITERIA
from judgement.prompt_names import LLM_SCORE_DRAFTS
from judgement.data.common.utils import read_file, get_chat_completion

class ScoreOutput(pydantic.BaseModel):
    SCORE: str
    REASONING: str

def generate_criteria(rough_draft_path: str, final_draft_path: str, criteria: str = "", model_type: str = GPT4_MINI, response_format: str = ScoreOutput):
    prompt = langfuse.get_prompt(
        LLM_SCORE_DRAFTS,
        type="chat"
    )
    
    # Load document pair
    rough_draft = read_file(rough_draft_path)
    final_draft = read_file(final_draft_path)
    
    raw_prompt = f"<ROUGH DRAFT>: {rough_draft}\n<FINAL DRAFT>: {final_draft}\n<CRITERIA>: {criteria}"
    
    compiled_prompt = prompt.compile(
        rough_draft=rough_draft,
        final_draft=final_draft,
        criteria=criteria
    )
    
    print(f"{compiled_prompt=}")
    
    # Format the response into two parts: the explanation for everything, and the criteria
    chat_completion = get_chat_completion(model_type, compiled_prompt, response_format=response_format)
    return raw_prompt, response_format.model_validate_json(chat_completion).model_dump()

if __name__ == "__main__":
    alma_documents_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_draft")
    alma_final_path = os.path.join(os.path.dirname(__file__), "documents/alma_anonymized_final")

    # Open a CSV file for writing
    with open("./judgement/eval/llm_score_drafts_dspy_data.csv", mode="w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["prompt", "score", "reasoning"])

        for draft_document_name in os.listdir(alma_documents_path):
            rough_draft_path = os.path.join(alma_documents_path, draft_document_name)
            final_draft_path = os.path.join(alma_final_path, draft_document_name)

            if os.path.exists(final_draft_path):
                prompt, output = generate_criteria(rough_draft_path, final_draft_path, BASE_LETTER_COMPARISON_CRITERIA)
                score = output["SCORE"]
                reasoning = output["REASONING"]
                
                # Write to CSV file
                csv_writer.writerow([prompt, score, reasoning])
            else:
                print(f"No corresponding file for: {draft_document_name}")
