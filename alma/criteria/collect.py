import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from judgement import *
from judgement.data.common import utils

alma = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
drafts = os.path.join(alma, 'alma_docs', 'alma_anonymized_draft')
finals = os.path.join(alma, 'alma_docs', 'alma_anonymized_final')
system_prompt_path = os.path.join(alma, 'criteria', 'system_prompt.txt')

letters = [f for f in os.listdir(drafts) if f.endswith('.txt')]

with open(system_prompt_path, 'r') as system_prompt_file:
    system_prompt = system_prompt_file.read()


prompt_template = "Please analyze and compare the following two documents related to an immigration case. Explain why the first document (Document A) is superior to the second document (Document B).\n\nDocument A:\n{final_text}\n\nDocument B:\n{draft_text}"

for draft_file in letters:
    draft_path = os.path.join(drafts, draft_file)
    final_path = os.path.join(finals, draft_file)
    
    with open(draft_path, 'r') as draft_f, open(final_path, 'r') as final_f:
        draft_text = draft_f.read()
        final_text = final_f.read()
    
    prompt_variables = {
        "final_text": final_text,
        "draft_text": draft_text,
    }
    
    formatted_prompt = prompt_template.format(**prompt_variables)

    model="LLAMA3_70B_INSTRUCT_TURBO"
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]

    response = utils.fetch_together_api_response(model, messages)
    
    # Write the response to a file in the alma_criteria folder
    criteria_folder = os.path.join(os.path.dirname(__file__), 'alma.criteria')
    os.makedirs(criteria_folder, exist_ok=True)
    output_file = os.path.join(criteria_folder, f"{os.path.splitext(draft_file)[0]}_criteria.txt")
    
    with open(output_file, 'w') as f:
        f.write(response)
    
    print(f"Criteria written to: {output_file}")
    print("\n" + "-"*50 + "\n")  # Separator between files
