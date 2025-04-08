from pydantic import BaseModel,EmailStr,Field
from typing import Optional


class Student(BaseModel):
    name : str = 'Jishnu'
    # name : str 
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0,lt=10)
# new_student = {'name' : 'Jishnu'}

# new_student = {'age':22}
new_student = {'age':'22','email':'jishnughosh@gmail.com','cgpa':8}

std = Student(**new_student)

# print(type(std))
# print(std.name)
print(std)
