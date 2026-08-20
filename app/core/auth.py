from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import hashlib
import base64
from app.core.config import settings
from app.database.connection import get_db
from app.database.models import Patient


# CONFIG


SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# auto_error=False → allows us to return 401 instead of FastAPI's 403
security = HTTPBearer(auto_error=False)


# TOKEN LOGIC


def create_access_token(data: dict) -> str:
    """
    Create a JWT access token with UTC-aware expiration
    """
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({
        "exp": expire,
        "sub": str(data.get("patient_id"))  # JWT best practice
    })

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Verify JWT token and return payload
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        patient_id = payload.get("patient_id")
        if patient_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

# CURRENT USER DEPENDENCY


async def get_current_patient(
    token_data: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db)
) -> Patient:
    """
    Fetch the logged-in patient from DB using JWT payload
    """

    patient_id = token_data.get("patient_id")

    result = await db.execute(
        select(Patient).where(
            Patient.id == patient_id
        )
    )

    patient = result.scalars().first()

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return patient

# PASSWORD LOGIC (bcrypt-safe, unlimited length)


def hash_password(password: str) -> str:
    """
    Secure password hashing:
    SHA-256 → Base64 → bcrypt
    (Solves bcrypt 72-byte limit)
    """
    sha256_bin = hashlib.sha256(password.encode("utf-8")).digest()
    b64_hash = base64.b64encode(sha256_bin).decode("utf-8")
    return pwd_context.hash(b64_hash)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against stored bcrypt hash
    """
    sha256_bin = hashlib.sha256(plain_password.encode("utf-8")).digest()
    b64_hash = base64.b64encode(sha256_bin).decode("utf-8")
    return pwd_context.verify(b64_hash, hashed_password)
