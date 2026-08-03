from pydantic import BaseModel, EmailStr, Field
from typing import Literal
from datetime import datetime, timezone


class UserInDB(BaseModel):
    """Shape of a user document as stored in MongoDB."""
    name: str
    email: EmailStr
    password_hash: str
    role: Literal["clinician", "admin", "developer"] = "clinician"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
