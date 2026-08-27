from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from datetime import datetime
from app.database.models import Patient

async def get_patient_by_email(
    db: AsyncSession,
    email: str,
):
    result = await db.execute(
        select(Patient).filter_by(email=email)
    )

    return result.scalars().first()



async def create_patient(
    db: AsyncSession,
    patient_data: dict,
    tumor_type: str,
    image_url: str,
    user_chat_thread: str,
    image_public_id: str,
):
    new_patient = Patient(
        username=patient_data['username'],
        email=patient_data['email'],
        password_hash=patient_data['password'], 
        city=patient_data['city'],
        state=patient_data['state'],
        country=patient_data['country'],
        tumor_type=tumor_type,
        mri_image_url=image_url,
        mri_image_public_id=image_public_id,
        user_chat_thread=user_chat_thread,
        created_at=datetime.utcnow(),
    )

    db.add(new_patient)
    await db.commit()
    await db.refresh(new_patient)
    return new_patient


    await db.execute(
        delete(ChatMessage).where(
        ChatMessage.patient_id == patient_id)
    )
    await db.commit()