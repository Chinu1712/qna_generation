import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, get_db
from models import Base, User, PDFDownload
from schemas import (
    SignupRequest,
    LoginRequest,
    UserResponse,
    LoginResponse,
    PDFDownloadResponse,
)
from auth import hash_password, verify_password

# Load .env from root qna folder
load_dotenv("../.env")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="QnA Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORED_PDF_FOLDER = "stored_pdfs"
os.makedirs(STORED_PDF_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Backend is running successfully"}


# -------------------------
# SIGNUP
# -------------------------
@app.post("/signup", response_model=UserResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing_username = db.query(User).filter(User.username == payload.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already exists")

    existing_email = db.query(User).filter(User.email == payload.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")

    new_user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


# -------------------------
# LOGIN
# -------------------------
@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        (User.username == payload.identifier) | (User.email == payload.identifier)
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User does not exist")

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    return {
        "message": "Login successful",
        "user": user
    }


# -------------------------
# GET ALL USERS
# -------------------------
@app.get("/users", response_model=list[UserResponse])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users


# -------------------------
# GET SINGLE USER
# -------------------------
@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# -------------------------
# SAVE GENERATED PDF
# -------------------------
@app.post("/save-pdf")
def save_pdf(
    user_id: int = Form(...),
    question_type: str = Form(...),
    pdf_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    safe_filename = f"user_{user_id}_{pdf_file.filename}"
    file_path = os.path.join(STORED_PDF_FOLDER, safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(pdf_file.file, buffer)

    new_pdf_record = PDFDownload(
        pdf_file_name=pdf_file.filename,
        pdf_file_path=file_path,
        question_type=question_type,
        user_id=user.id
    )

    db.add(new_pdf_record)
    db.commit()
    db.refresh(new_pdf_record)

    return {
        "message": "PDF saved successfully",
        "pdf_id": new_pdf_record.id,
        "pdf_file_name": new_pdf_record.pdf_file_name,
        "pdf_file_path": new_pdf_record.pdf_file_path,
        "question_type": new_pdf_record.question_type
    }


# -------------------------
# GET PDF HISTORY OF USER
# -------------------------
@app.get("/user-pdfs/{user_id}", response_model=list[PDFDownloadResponse])
def get_user_pdfs(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user.pdf_downloads


# -------------------------
# DELETE PDF RECORD
# -------------------------
@app.delete("/delete-pdf/{pdf_id}")
def delete_pdf(pdf_id: int, db: Session = Depends(get_db)):
    pdf_record = db.query(PDFDownload).filter(PDFDownload.id == pdf_id).first()
    if not pdf_record:
        raise HTTPException(status_code=404, detail="PDF record not found")

    if os.path.exists(pdf_record.pdf_file_path):
        os.remove(pdf_record.pdf_file_path)

    db.delete(pdf_record)
    db.commit()

    return {"message": "PDF record deleted successfully"}