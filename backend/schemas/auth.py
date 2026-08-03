from pydantic import BaseModel, EmailStr
from typing import Literal


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Literal["clinician", "admin", "developer"] = "clinician"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: str


class LoginResponse(BaseModel):
    token: str
    user: UserOut
