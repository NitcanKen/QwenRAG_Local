"""
Customer management endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.models.customer import Customer, CustomerCreate, CustomerUpdate
from app.core.database import DatabaseManager
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def list_customers(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, le=1000),
    gender: Optional[str] = Query(default=None),
    contract: Optional[str] = Query(default=None),
    churn: Optional[int] = Query(default=None),
):
    """
    List customers with optional filtering.
    
    Args:
        skip: Number of customers to skip
        limit: Maximum number of customers to return
        gender: Filter by gender
        contract: Filter by contract type
        churn: Filter by churn status (0 or 1)
    """
    try:
        logger.info(f"Listing customers with filters: gender={gender}, contract={contract}, churn={churn}")
        
        # Build filters dictionary
        filters = {}
        if gender is not None:
            filters["gender"] = gender
        if contract is not None:
            filters["contract"] = contract
        if churn is not None:
            filters["churn"] = churn
        
        # Get customers from database
        customers = await DatabaseManager.get_customers(
            skip=skip,
            limit=limit,
            filters=filters if filters else None
        )
        
        return JSONResponse(
            content={
                "customers": customers,
                "count": len(customers),
                "skip": skip,
                "limit": limit,
                "filters": filters
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing customers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve customers: {str(e)}")


@router.get("/{customer_id}")
async def get_customer(customer_id: str):
    """
    Get a specific customer by ID.
    
    Args:
        customer_id: Customer ID to retrieve
    """
    try:
        logger.info(f"Getting customer: {customer_id}")
        
        # Get customer from database
        customer = await DatabaseManager.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        return JSONResponse(content={"customer": customer})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve customer: {str(e)}")


@router.post("/")
async def create_customer(customer: CustomerCreate):
    """
    Create a new customer.
    
    Args:
        customer: Customer data to create
    """
    try:
        logger.info(f"Creating customer: {customer.customer_id}")
        
        # Convert Pydantic model to dict, excluding None values
        customer_data = customer.model_dump(exclude_none=True)
        
        # Create customer in database
        created_customer = await DatabaseManager.create_customer(customer_data)
        
        return JSONResponse(
            content={
                "message": "Customer created successfully",
                "customer": created_customer
            },
            status_code=201
        )
        
    except Exception as e:
        logger.error(f"Error creating customer {customer.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create customer: {str(e)}")


@router.put("/{customer_id}")
async def update_customer(customer_id: str, customer: CustomerUpdate):
    """
    Update an existing customer.
    
    Args:
        customer_id: Customer ID to update
        customer: Updated customer data
    """
    try:
        logger.info(f"Updating customer: {customer_id}")
        
        # Convert Pydantic model to dict, excluding None values
        update_data = customer.model_dump(exclude_none=True)
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update provided")
        
        # Update customer in database
        updated_customer = await DatabaseManager.update_customer(customer_id, update_data)
        
        if not updated_customer:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
        
        return JSONResponse(
            content={
                "message": "Customer updated successfully",
                "customer": updated_customer
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update customer: {str(e)}")