"""
This script demonstrates how to use Pydantic's BaseModel for data validation.
It shows examples of default values, optional fields, and built-in validators.
Email validation and value constraints (like cgpa between 0 and 10) are
included.  It also covers setting custom field descriptions using Field.
Finally, the model is converted to dictionary and JSON format.
Coerce means Pydantic helps fix the data type for you, even if you gave it the
"wrong" type.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):

    name: str


# new_student = {'name': 32} --> it will raise error
new_student = {"name": "nitish"}

student = Student(**new_student)

print(type(student))

# set default value


class Student(BaseModel):

    name: str = "nitish"


new_student = {}

student = Student(**new_student)

print(type(student))

# optional fields


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None


new_student = {"age": 32}

student = Student(**new_student)

print(student)

# Builtin validation


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr


# new_student = {'age': 32, 'email': 'abc'}     -- it raise error
new_student = {"age": 32, "email": "abc@gmail.com"}

student = Student(**new_student)

print(student)

# Field functions --> default values, constraints, description, regex

# constraints(cgpa should be in the range of 0-10)


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)


# new_student = {'age': 32, 'email': 'abc@gmail.com', 'cgpa': 12}
# --> it will raise error
new_student = {"age": 32, "email": "abc@gmail.com", "cgpa": 5}


student = Student(**new_student)

print(student)

# default value


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5)


new_student = {"age": 32, "email": "abc@gmail.com"}


student = Student(**new_student)

print(student)

# custom description


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of the student",
    )


new_student = {"age": 32, "email": "abc@gmail.com"}


student = Student(**new_student)

print(student)

# Returns pydantic object --> convert to json/dict


class Student(BaseModel):

    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of the student",
    )


new_student = {"age": 32, "email": "abc@gmail.com"}


student = Student(**new_student)

student_dict = dict(student)

print(student_dict["age"])

student_json = student.model_dump_json()
