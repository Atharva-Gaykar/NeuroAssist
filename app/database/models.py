from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.config import settings
from app.database.connection import engine
from sqlalchemy.dialects.postgresql import UUID
import uuid


Base=declarative_base()

 # adjust import to your actual Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(60), nullable=False)  # bcrypt/argon2 hash only, never plaintext

    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    tumor_type = Column(String(100), nullable=False)

    # MRI stored in Cloudinary/object storage, not in the DB
    mri_image_url = Column(String(500), nullable=True)
    mri_image_public_id = Column(String(255), nullable=True)  # e.g. Cloudinary public_id, for deletion/updates

    # One thread per patient for persistent conversational context
    user_chat_thread = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)



def create_tables():
    Base.metadata.create_all(bind=engine)


# if __name__ == "__main__":
#     create_tables()
#     print("Tables created successfully")




