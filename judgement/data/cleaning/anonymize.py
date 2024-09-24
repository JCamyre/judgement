"""
Takes a document and anonymizes it by replacing all names with alternative names.
"""

from dotenv import load_dotenv
from openai import OpenAI

# Load the environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

def replace_identifying_info(document):
    # Define the message for the GPT-4 model
    messages = [
        {
            "role": "user",
            "content": f"""
You are a helpful assistant. Please anonymize the following document by replacing all identifying information such as names, places, companies, etc. with alternatives:

Document:
{document}

Anonymized Document:
"""
        }
    ]
    
    # Create the completion request using the chat endpoint
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Extract the anonymized document from the response
    print(completion)
    anonymized_document = completion.choices[0].message
    
    return anonymized_document

if __name__ == "__main__":
    document = """
    Dear Mr. John Doe,

    Thank you for your email regarding the project at Acme Corp. As you mentioned, the meeting in New York was very productive...

    Sincerely,
    Jane Smith
    """

    anonymized_document = replace_identifying_info(document)
    print(anonymized_document)
