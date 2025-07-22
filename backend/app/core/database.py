"""
Database configuration and connection management.
"""

import asyncio
from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, DateTime, UUID as SQLAlchemyUUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from supabase import create_client, Client
import uuid
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Database engines
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None

# Supabase client
supabase: Optional[Client] = None


class CustomerTable(Base):
    """SQLAlchemy model for customers table."""
    __tablename__ = "customers"
    
    id = Column(SQLAlchemyUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(String, unique=True, nullable=False, index=True)
    gender = Column(String)
    senior_citizen = Column(Integer)
    partner = Column(String)
    dependents = Column(String)
    tenure = Column(Integer)
    phone_service = Column(String)
    multiple_lines = Column(String)
    internet_service = Column(String)
    online_security = Column(String)
    online_backup = Column(String)
    device_protection = Column(String)
    tech_support = Column(String)
    streaming_tv = Column(String)
    streaming_movies = Column(String)
    contract = Column(String)
    paperless_billing = Column(String)
    payment_method = Column(String)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    churn = Column(Integer)
    tenure_group = Column(String)
    monthly_charges_group = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def init_supabase() -> Optional[Client]:
    """
    Initialize Supabase client.
    
    Returns:
        Supabase client instance or None if configuration is missing
    """
    global supabase
    
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            logger.warning("Supabase configuration missing, falling back to PostgreSQL")
            return None
            
        supabase = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        
        # Test connection
        response = supabase.table("customers").select("count").limit(1).execute()
        logger.info("✅ Supabase connection established successfully")
        return supabase
        
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
        return None


def init_postgresql() -> bool:
    """
    Initialize PostgreSQL connection as fallback.
    
    Returns:
        True if successful, False otherwise
    """
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    try:
        if not settings.DATABASE_URL:
            logger.warning("No DATABASE_URL configured for PostgreSQL fallback")
            return False
            
        # Synchronous engine
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Asynchronous engine (convert postgresql:// to postgresql+asyncpg://)
        async_db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        async_engine = create_async_engine(
            async_db_url,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Session makers
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        AsyncSessionLocal = async_sessionmaker(
            async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test connection
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            logger.info("✅ PostgreSQL connection established successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL: {e}")
        return False


async def init_database():
    """
    Initialize database connections.
    Try Supabase first, fallback to PostgreSQL.
    """
    logger.info("Initializing database connections...")
    
    # Try Supabase first
    supabase_client = init_supabase()
    if supabase_client:
        logger.info("Using Supabase as primary database")
        return
    
    # Fallback to PostgreSQL
    if init_postgresql():
        logger.info("Using PostgreSQL as fallback database")
        
        # Create tables if they don't exist
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
    else:
        logger.error("No database connection available! Running in fallback mode.")
        # Don't raise exception - allow server to start without database


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        Database session
    """
    if supabase:
        # For Supabase, we don't use SQLAlchemy sessions
        # This is a placeholder - actual Supabase operations will use the client directly
        yield None
    elif AsyncSessionLocal:
        async with AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
    else:
        raise Exception("No database connection available")


def get_supabase_client() -> Client:
    """
    Get Supabase client instance.
    
    Returns:
        Supabase client
        
    Raises:
        Exception: If Supabase is not initialized
    """
    if not supabase:
        raise Exception("Supabase client not initialized")
    return supabase


class DatabaseManager:
    """Database operations manager that works with both Supabase and PostgreSQL."""
    
    @staticmethod
    async def get_customers(
        skip: int = 0,
        limit: int = 100,
        filters: Optional[dict] = None
    ) -> list:
        """
        Get customers with optional filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters dictionary
            
        Returns:
            List of customer records
        """
        if supabase:
            # Supabase query
            query = supabase.table("customers").select("*")
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        query = query.eq(key, value)
            
            # Apply pagination
            query = query.range(skip, skip + limit - 1)
            
            response = query.execute()
            return response.data
            
        elif AsyncSessionLocal:
            # PostgreSQL query
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                
                query = select(CustomerTable).offset(skip).limit(limit)
                
                # Apply filters
                if filters:
                    for key, value in filters.items():
                        if value is not None and hasattr(CustomerTable, key):
                            query = query.where(getattr(CustomerTable, key) == value)
                
                result = await session.execute(query)
                customers = result.scalars().all()
                
                # Convert to dict format
                return [
                    {
                        "id": str(customer.id),
                        "customer_id": customer.customer_id,
                        "gender": customer.gender,
                        "senior_citizen": customer.senior_citizen,
                        "partner": customer.partner,
                        "dependents": customer.dependents,
                        "tenure": customer.tenure,
                        "phone_service": customer.phone_service,
                        "multiple_lines": customer.multiple_lines,
                        "internet_service": customer.internet_service,
                        "online_security": customer.online_security,
                        "online_backup": customer.online_backup,
                        "device_protection": customer.device_protection,
                        "tech_support": customer.tech_support,
                        "streaming_tv": customer.streaming_tv,
                        "streaming_movies": customer.streaming_movies,
                        "contract": customer.contract,
                        "paperless_billing": customer.paperless_billing,
                        "payment_method": customer.payment_method,
                        "monthly_charges": customer.monthly_charges,
                        "total_charges": customer.total_charges,
                        "churn": customer.churn,
                        "tenure_group": customer.tenure_group,
                        "monthly_charges_group": customer.monthly_charges_group,
                        "created_at": customer.created_at.isoformat() if customer.created_at else None,
                        "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
                    }
                    for customer in customers
                ]
        else:
            raise Exception("No database connection available")
    
    @staticmethod
    async def create_customer(customer_data: dict) -> dict:
        """
        Create a new customer.
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Created customer record
        """
        if supabase:
            response = supabase.table("customers").insert(customer_data).execute()
            return response.data[0]
            
        elif AsyncSessionLocal:
            async with AsyncSessionLocal() as session:
                # Create new customer instance
                customer = CustomerTable(**customer_data)
                session.add(customer)
                await session.commit()
                await session.refresh(customer)
                
                return {
                    "id": str(customer.id),
                    "customer_id": customer.customer_id,
                    "gender": customer.gender,
                    "senior_citizen": customer.senior_citizen,
                    "partner": customer.partner,
                    "dependents": customer.dependents,
                    "tenure": customer.tenure,
                    "phone_service": customer.phone_service,
                    "multiple_lines": customer.multiple_lines,
                    "internet_service": customer.internet_service,
                    "online_security": customer.online_security,
                    "online_backup": customer.online_backup,
                    "device_protection": customer.device_protection,
                    "tech_support": customer.tech_support,
                    "streaming_tv": customer.streaming_tv,
                    "streaming_movies": customer.streaming_movies,
                    "contract": customer.contract,
                    "paperless_billing": customer.paperless_billing,
                    "payment_method": customer.payment_method,
                    "monthly_charges": customer.monthly_charges,
                    "total_charges": customer.total_charges,
                    "churn": customer.churn,
                    "tenure_group": customer.tenure_group,
                    "monthly_charges_group": customer.monthly_charges_group,
                    "created_at": customer.created_at.isoformat() if customer.created_at else None,
                    "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
                }
        else:
            raise Exception("No database connection available")
    
    @staticmethod
    async def update_customer(customer_id: str, update_data: dict) -> Optional[dict]:
        """
        Update an existing customer.
        
        Args:
            customer_id: Customer ID to update
            update_data: Fields to update
            
        Returns:
            Updated customer record or None
        """
        if supabase:
            response = supabase.table("customers").update(update_data).eq("customer_id", customer_id).execute()
            return response.data[0] if response.data else None
            
        elif AsyncSessionLocal:
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                
                # Get existing customer
                query = select(CustomerTable).where(CustomerTable.customer_id == customer_id)
                result = await session.execute(query)
                customer = result.scalar_one_or_none()
                
                if not customer:
                    return None
                
                # Update fields
                for key, value in update_data.items():
                    if hasattr(customer, key):
                        setattr(customer, key, value)
                
                await session.commit()
                await session.refresh(customer)
                
                return {
                    "id": str(customer.id),
                    "customer_id": customer.customer_id,
                    "gender": customer.gender,
                    "senior_citizen": customer.senior_citizen,
                    "partner": customer.partner,
                    "dependents": customer.dependents,
                    "tenure": customer.tenure,
                    "phone_service": customer.phone_service,
                    "multiple_lines": customer.multiple_lines,
                    "internet_service": customer.internet_service,
                    "online_security": customer.online_security,
                    "online_backup": customer.online_backup,
                    "device_protection": customer.device_protection,
                    "tech_support": customer.tech_support,
                    "streaming_tv": customer.streaming_tv,
                    "streaming_movies": customer.streaming_movies,
                    "contract": customer.contract,
                    "paperless_billing": customer.paperless_billing,
                    "payment_method": customer.payment_method,
                    "monthly_charges": customer.monthly_charges,
                    "total_charges": customer.total_charges,
                    "churn": customer.churn,
                    "tenure_group": customer.tenure_group,
                    "monthly_charges_group": customer.monthly_charges_group,
                    "created_at": customer.created_at.isoformat() if customer.created_at else None,
                    "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
                }
        else:
            raise Exception("No database connection available")

    @staticmethod
    async def get_customer_by_id(customer_id: str) -> Optional[dict]:
        """
        Get a customer by ID.
        
        Args:
            customer_id: Customer ID to retrieve
            
        Returns:
            Customer record or None
        """
        if supabase:
            response = supabase.table("customers").select("*").eq("customer_id", customer_id).execute()
            return response.data[0] if response.data else None
            
        elif AsyncSessionLocal:
            async with AsyncSessionLocal() as session:
                from sqlalchemy import select
                
                query = select(CustomerTable).where(CustomerTable.customer_id == customer_id)
                result = await session.execute(query)
                customer = result.scalar_one_or_none()
                
                if customer:
                    return {
                        "id": str(customer.id),
                        "customer_id": customer.customer_id,
                        "gender": customer.gender,
                        "senior_citizen": customer.senior_citizen,
                        "partner": customer.partner,
                        "dependents": customer.dependents,
                        "tenure": customer.tenure,
                        "phone_service": customer.phone_service,
                        "multiple_lines": customer.multiple_lines,
                        "internet_service": customer.internet_service,
                        "online_security": customer.online_security,
                        "online_backup": customer.online_backup,
                        "device_protection": customer.device_protection,
                        "tech_support": customer.tech_support,
                        "streaming_tv": customer.streaming_tv,
                        "streaming_movies": customer.streaming_movies,
                        "contract": customer.contract,
                        "paperless_billing": customer.paperless_billing,
                        "payment_method": customer.payment_method,
                        "monthly_charges": customer.monthly_charges,
                        "total_charges": customer.total_charges,
                        "churn": customer.churn,
                        "tenure_group": customer.tenure_group,
                        "monthly_charges_group": customer.monthly_charges_group,
                        "created_at": customer.created_at.isoformat() if customer.created_at else None,
                        "updated_at": customer.updated_at.isoformat() if customer.updated_at else None,
                    }
                return None
        else:
            raise Exception("No database connection available")


# Initialize database on module import (will be called by FastAPI startup)
database_manager = DatabaseManager()