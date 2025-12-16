import os
from datetime import timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# App metadata and initialization with OpenAPI tags
app = FastAPI(
    title="Notes API",
    description="FastAPI backend for a personal notes app with JWT authentication and CRUD endpoints.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User registration and login"},
        {"name": "notes", "description": "CRUD operations for notes (protected)"},
        {"name": "health", "description": "Health and meta endpoints"},
    ],
)

# CORS configuration - restrict to frontend origin
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
if not DB_URL:
    # Provide a sane default for local development (commented guideline)
    # Example: postgres://user:password@localhost:5432/notes_database
    # We will raise at startup if missing to ensure explicit configuration
    pass

# Create SQLAlchemy engine and session
engine = create_engine(DB_URL, pool_pre_ping=True) if DB_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()

# Auth configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
    scheme_name="JWT"
)

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    notes = relationship("Note", back_populates="owner", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="notes")


# Pydantic Schemas
class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Token type, typically 'bearer'")


class TokenData(BaseModel):
    user_id: Optional[int] = Field(None, description="User ID extracted from token")


class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")


class UserOut(BaseModel):
    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")

    class Config:
        from_attributes = True


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, description="Note title")
    content: Optional[str] = Field(None, description="Note content")


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, description="New title")
    content: Optional[str] = Field(None, description="New content")


class NoteOut(BaseModel):
    id: int = Field(..., description="Note ID")
    title: str = Field(..., description="Note title")
    content: Optional[str] = Field(None, description="Note content")
    user_id: int = Field(..., description="Owner user ID")

    class Config:
        from_attributes = True


# Database dependency
def get_db():
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL/POSTGRES_URL not configured. See .env.example.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Security helpers
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": (timedelta(seconds=0) + expire).total_seconds()})  # placeholder; we will set exp with datetime below
    # Proper expiration using datetime
    from datetime import datetime as _dt, timezone as _tz
    expire_dt = _dt.now(_tz.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire_dt
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# PUBLIC_INTERFACE
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Retrieve the current user from JWT token.

    This function validates the JWT bearer token, extracts the user ID, and returns the corresponding User model.

    Raises:
        HTTPException 401 if the token is invalid or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not SECRET_KEY:
        raise HTTPException(status_code=500, detail="JWT_SECRET_KEY not configured.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=int(user_id))
    except JWTError:
        raise credentials_exception
    user = db.get(User, token_data.user_id)
    if user is None:
        raise credentials_exception
    return user


# Routes

@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Health check endpoint to verify the API is up."""
    return {"message": "Healthy"}


# Auth endpoints
@app.post("/auth/register", response_model=UserOut, tags=["auth"], summary="Register a new user")
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account with email and password.

    Parameters:
        payload: UserCreate schema with email and password

    Returns:
        UserOut: created user data (without password)

    Errors:
        400 if email already exists.
    """
    existing = get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = get_password_hash(payload.password)
    user = User(email=payload.email, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token, tags=["auth"], summary="Login and obtain JWT")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate a user and return a JWT access token.

    Parameters:
        form_data: OAuth2PasswordRequestForm containing 'username' (email) and 'password'

    Returns:
        Token: JWT bearer token

    Errors:
        400 for incorrect email or password.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": str(user.id)}, expires_delta=access_token_expires)
    return {"access_token": token, "token_type": "bearer"}


# Notes endpoints (protected)
@app.get("/notes", response_model=List[NoteOut], tags=["notes"], summary="List notes")
def list_notes(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all notes for the authenticated user."""
    notes = db.query(Note).filter(Note.user_id == current_user.id).order_by(Note.created_at.desc()).all()
    return notes


@app.post("/notes", response_model=NoteOut, status_code=201, tags=["notes"], summary="Create note")
def create_note(payload: NoteCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new note for the authenticated user."""
    note = Note(title=payload.title, content=payload.content, user_id=current_user.id)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@app.get("/notes/{note_id}", response_model=NoteOut, tags=["notes"], summary="Get note by ID")
def get_note(note_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get a specific note by ID for the authenticated user."""
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@app.put("/notes/{note_id}", response_model=NoteOut, tags=["notes"], summary="Update note by ID")
def update_note(note_id: int, payload: NoteUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update title/content for a specific note (owned by user)."""
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")
    if payload.title is not None:
        if not payload.title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        note.title = payload.title
    if payload.content is not None:
        note.content = payload.content
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@app.delete("/notes/{note_id}", status_code=204, tags=["notes"], summary="Delete note by ID")
def delete_note(note_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a specific note owned by the authenticated user."""
    note = db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")
    db.delete(note)
    db.commit()
    return None


# Startup to create tables
@app.on_event("startup")
def on_startup():
    """Create database tables on startup if they do not exist."""
    if engine is None:
        raise RuntimeError("DATABASE_URL/POSTGRES_URL not configured. See .env.example.")
    Base.metadata.create_all(bind=engine)
