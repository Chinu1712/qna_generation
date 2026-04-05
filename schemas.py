from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


class SignupRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    identifier: str   # username or email
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    message: str
    user: UserResponse


class PDFDownloadResponse(BaseModel):
    id: int
    pdf_file_name: str
    pdf_file_path: str
    question_type: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class UserPDFHistoryResponse(BaseModel):
    id: int
    username: str
    email: str
    pdf_downloads: List[PDFDownloadResponse] = []

    class Config:
        from_attributes = True