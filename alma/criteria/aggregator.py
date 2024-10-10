import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from judgement import *
from judgement.data.common import utils

# Define the path to the alma_criteria folder
alma_criteria_folder = os.path.join(os.path.dirname(__file__), 'alma_criteria')

aggregate_prompt_template = "Please analyze the following set of criteria used to compare legal documents in immigration cases. Your task is to synthesize these criteria into a concise, comprehensive list of essential evaluation points. Focus on identifying recurring themes and the most significant factors that contribute to document quality and effectiveness.\n\nCriteria set {i}:\n{criteria_content}"
# Set up the model and messages for the API call
model = "LLAMA3_70B_INSTRUCT_TURBO"
messages = [
    {"role": "system", "content": open(os.path.join(os.path.dirname(__file__), 'aggregator_prompt.txt'), 'r').read()}
]

# Iterate through all files in the alma_criteria folder
for i, filename in enumerate(os.listdir(alma_criteria_folder), 1):
    if filename.endswith('.txt'):
        file_path = os.path.join(alma_criteria_folder, filename)
        
        with open(file_path, 'r') as file:
            content = file.read()
        

        # Prepare the prompt with the content of the current file
        prompt_variables = {
            "i": i,
            "criteria_content": content
        }
        
        formatted_prompt = aggregate_prompt_template.format(**prompt_variables)

        messages.append({"role": "user", "content": formatted_prompt})


# Make the API call
response = utils.fetch_together_api_response(model, messages)

# Write the aggregated criteria to a new file
output_filename = "criteria.txt"
output_path = os.path.join(os.path.dirname(__file__), output_filename)

with open(output_path, 'w') as output_file:
    output_file.write(response)

print(f"Aggregated criteria written to: {output_path}")
print("\n" + "-"*50 + "\n")  # Separator between files










