"""
Customer data models.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class CustomerBase(BaseModel):
    """Base customer model with common fields."""
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: Optional[str] = Field(None, description="Customer gender")
    senior_citizen: Optional[int] = Field(None, description="Senior citizen status (0/1)")
    partner: Optional[str] = Field(None, description="Has partner (Yes/No)")
    dependents: Optional[str] = Field(None, description="Has dependents (Yes/No)")
    tenure: Optional[int] = Field(None, description="Months with company")
    phone_service: Optional[str] = Field(None, description="Has phone service (Yes/No)")
    multiple_lines: Optional[str] = Field(None, description="Has multiple lines")
    internet_service: Optional[str] = Field(None, description="Internet service type")
    online_security: Optional[str] = Field(None, description="Has online security")
    online_backup: Optional[str] = Field(None, description="Has online backup")
    device_protection: Optional[str] = Field(None, description="Has device protection")
    tech_support: Optional[str] = Field(None, description="Has tech support")
    streaming_tv: Optional[str] = Field(None, description="Has streaming TV")
    streaming_movies: Optional[str] = Field(None, description="Has streaming movies")
    contract: Optional[str] = Field(None, description="Contract type")
    paperless_billing: Optional[str] = Field(None, description="Has paperless billing")
    payment_method: Optional[str] = Field(None, description="Payment method")
    monthly_charges: Optional[float] = Field(None, description="Monthly charges amount")
    total_charges: Optional[float] = Field(None, description="Total charges amount")
    churn: Optional[int] = Field(None, description="Churn status (0/1)")
    tenure_group: Optional[str] = Field(None, description="Tenure group classification")
    monthly_charges_group: Optional[str] = Field(None, description="Monthly charges group")


class CustomerCreate(CustomerBase):
    """Model for creating a new customer."""
    pass


class CustomerUpdate(BaseModel):
    """Model for updating an existing customer."""
    gender: Optional[str] = None
    senior_citizen: Optional[int] = None
    partner: Optional[str] = None
    dependents: Optional[str] = None
    tenure: Optional[int] = None
    phone_service: Optional[str] = None
    multiple_lines: Optional[str] = None
    internet_service: Optional[str] = None
    online_security: Optional[str] = None
    online_backup: Optional[str] = None
    device_protection: Optional[str] = None
    tech_support: Optional[str] = None
    streaming_tv: Optional[str] = None
    streaming_movies: Optional[str] = None
    contract: Optional[str] = None
    paperless_billing: Optional[str] = None
    payment_method: Optional[str] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    churn: Optional[int] = None
    tenure_group: Optional[str] = None
    monthly_charges_group: Optional[str] = None


class Customer(CustomerBase):
    """Full customer model with database fields."""
    id: UUID = Field(..., description="Database UUID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True