from pydantic import BaseModel, Field
class VectorSearchInput(BaseModel):
    query: str = Field(description="Standalone, self-contained search query about the brain tumor topic.")



class SerpSearchInput(BaseModel):
    query: str = Field(description="Standalone, self-contained search query about the brain tumor topic.")