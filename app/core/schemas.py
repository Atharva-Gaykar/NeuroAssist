from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class TokenResponse(BaseModel):
    success: bool
    token: str
    message: str