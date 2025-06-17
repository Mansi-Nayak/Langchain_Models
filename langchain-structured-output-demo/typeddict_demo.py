"""
This script demonstrates the use of Python's TypedDict from the typing module
to define a structured dictionary type named 'Person'. The 'Person' type expects
two fields:
- 'name' (of type str)
- 'age' (of type int)
"""

from typing import TypedDict


class Person(TypedDict):

    name: str
    age: int


new_person: Person = {"name": "nitish", "age": "35"}

print(new_person)
