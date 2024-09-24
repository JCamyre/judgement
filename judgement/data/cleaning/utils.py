"""
This file contains utility functions used in data cleaning scripts
"""

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

