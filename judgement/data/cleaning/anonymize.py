"""
Takes a document and anonymizes it by replacing all names with alternative names.
"""
import os
import pprint
from dotenv import load_dotenv
from openai import OpenAI

# Load the environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

PROMPT = [
    {
        "role": "system",
        "content": "You are an AI assistant. Your task is to anonymize documents by replacing all identifying information such as names, places, companies, etc., with alternative, non-identifying information. Replace names with other common names, places with other cities or regions, and companies with other generic or fictional company names. Ensure that the replacements make sense contextually. Gender of names does not need to match; feel free to replace male names with female names and vice versa. The output should read naturally as if it were the original content."
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Dear Mr. John Doe, Thank you for your email regarding the project at Acme Corp. As you mentioned, the meeting in New York was very productive. Sincerely, Jane Smith"
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Dear Mr. Andrew Lee, Thank you for your email regarding the project at Beta Solutions. As you mentioned, the meeting in Chicago was very productive. Sincerely, Rebecca Taylor"
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Sarah and Michael attended the conference at Google in Mountain View. They later visited the Golden Gate Bridge in San Francisco."
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Emily and James attended the conference at Innovatech in Seattle. They later visited the Space Needle in Portland."
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Tom received a job offer from IBM after graduating from Stanford University."
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Lucy received a job offer from Apex Technologies after graduating from Columbia University."
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "The contract was signed by David on behalf of Microsoft and delivered to the office in Redmond."
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "The contract was signed by Sophia on behalf of GreenTech and delivered to the office in Boston."
    }
]

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def replace_identifying_info(document_path):
    # Define the message for the GPT-4 model
    messages = [
        {
            "role": "user",
            "content": read_file(document_path)
        }
    ]
    pprint.pprint(messages)
    # Create the completion request using the chat endpoint
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=PROMPT + messages
    )
    
    # Extract the anonymized document from the response
    # print(completion)
    anonymized_document = completion.choices[0].message.content
    
    return anonymized_document

if __name__ == "__main__":
    document = os.path.join(os.path.dirname(__file__), "samples", "example_letter.txt")

    anonymized_document = replace_identifying_info(document)
    print("*" * 50)
    print(anonymized_document)
