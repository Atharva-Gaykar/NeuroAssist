from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,      
    pool_recycle=300,          
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


pool = AsyncConnectionPool(
    conninfo=settings.LANGGRAPH_DB_URL,
    min_size=1,
    max_size=5,
    open=False, # We open it manually on the next line
    check=AsyncConnectionPool.check_connection,
    kwargs={"autocommit": True, "row_factory": dict_row}
)


async def get_db():
    async with SessionLocal() as db:
        yield db




# I encountered the greenlet_spawn error because I initially didn't use expire_on_commit=False and class_=AsyncSession. 
# In async SQLAlchemy, any automatic database access—like lazy loading or reloading expired objects after a commit—fails because it attempts synchronous I/O in an async environment. 
# Configuring the session maker properly keeps the data cached in memory and prevents those unexpected, un-awaited database calls