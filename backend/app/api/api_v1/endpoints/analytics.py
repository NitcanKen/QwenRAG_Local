"""
Analytics endpoints for churn analysis.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.analytics import analytics_service
from app.core.logging import get_logger
from app.core.exceptions import DatabaseConnectionError

logger = get_logger(__name__)
router = APIRouter()


@router.get("/churn-rate")
async def get_churn_rate() -> Dict[str, Any]:
    """
    Get overall churn rate metrics.
    
    Returns:
        Dictionary containing churn rate statistics
    """
    try:
        logger.info("Getting overall churn rate metrics")
        metrics = await analytics_service.get_overall_churn_metrics()
        
        return JSONResponse(content=metrics)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in churn rate endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in churn rate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate churn rate metrics")


@router.get("/churn-by-tenure")
async def get_churn_by_tenure() -> Dict[str, Any]:
    """
    Get churn analysis by tenure groups.
    
    Returns:
        Dictionary containing churn analysis by tenure
    """
    try:
        logger.info("Getting churn analysis by tenure")
        analysis = await analytics_service.get_churn_rate_by_tenure()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in tenure analysis endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in tenure analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate tenure analysis")


@router.get("/churn-by-contract")
async def get_churn_by_contract() -> Dict[str, Any]:
    """
    Get churn analysis by contract type.
    
    Returns:
        Dictionary containing churn analysis by contract
    """
    try:
        logger.info("Getting churn analysis by contract")
        analysis = await analytics_service.get_churn_rate_by_contract()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in contract analysis endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in contract analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate contract analysis")


@router.get("/churn-by-payment-method")
async def get_churn_by_payment_method() -> Dict[str, Any]:
    """
    Get churn analysis by payment method.
    
    Returns:
        Dictionary containing churn analysis by payment method
    """
    try:
        logger.info("Getting churn analysis by payment method")
        analysis = await analytics_service.get_churn_rate_by_payment_method()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in payment method analysis endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in payment method analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate payment method analysis")


@router.get("/demographics")
async def get_demographic_analysis() -> Dict[str, Any]:
    """
    Get demographic analysis of churn patterns.
    
    Returns:
        Dictionary containing demographic churn analysis
    """
    try:
        logger.info("Getting demographic churn analysis")
        analysis = await analytics_service.get_demographic_analysis()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in demographic analysis endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in demographic analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate demographic analysis")


@router.get("/services")
async def get_service_impact_analysis() -> Dict[str, Any]:
    """
    Get service impact analysis on churn.
    
    Returns:
        Dictionary containing service impact analysis
    """
    try:
        logger.info("Getting service impact churn analysis")
        analysis = await analytics_service.get_service_impact_analysis()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in service impact analysis endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in service impact analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate service impact analysis")


@router.get("/financial")
async def get_financial_metrics() -> Dict[str, Any]:
    """
    Get financial metrics related to churn.
    
    Returns:
        Dictionary containing financial analysis
    """
    try:
        logger.info("Getting financial churn metrics")
        analysis = await analytics_service.get_financial_metrics()
        
        return JSONResponse(content=analysis)
        
    except DatabaseConnectionError as e:
        logger.error(f"Database error in financial metrics endpoint: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in financial metrics endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate financial metrics")