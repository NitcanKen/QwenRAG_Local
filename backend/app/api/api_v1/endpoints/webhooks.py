"""
Webhook endpoints for external services.
"""

import hmac
import hashlib
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse

from app.services.realtime import realtime_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def verify_supabase_webhook(request: Request, signature: str = None) -> bool:
    """
    Verify Supabase webhook signature.
    
    Args:
        request: FastAPI request object
        signature: Webhook signature header
        
    Returns:
        True if signature is valid
    """
    try:
        if not signature:
            logger.warning("No signature provided for webhook verification")
            return False
        
        # Get raw body
        body = await request.body()
        
        # In production, you would use your actual webhook secret
        webhook_secret = getattr(settings, 'SUPABASE_WEBHOOK_SECRET', 'development-webhook-secret')
        
        # Calculate expected signature
        expected_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
        
    except Exception as e:
        logger.error(f"Error verifying webhook signature: {e}")
        return False


@router.post("/supabase")
async def supabase_webhook_handler(
    request: Request,
    x_signature: str = Header(None, alias="x-signature")
) -> Dict[str, Any]:
    """
    Handle Supabase database change webhooks.
    
    Args:
        request: FastAPI request object
        x_signature: Webhook signature header
        
    Returns:
        Acknowledgment response
    """
    try:
        logger.info("Received Supabase webhook")
        
        # Verify webhook signature in production
        if settings.ENVIRONMENT == "production":
            if not await verify_supabase_webhook(request, x_signature):
                logger.warning("Invalid webhook signature")
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse webhook payload
        payload = await request.json()
        logger.info(f"Webhook payload: {payload}")
        
        # Extract change information
        change_event = {
            "type": payload.get("type", "UNKNOWN"),
            "table": payload.get("table", "unknown"),
            "record": payload.get("record", {}),
            "old_record": payload.get("old_record", {}),
            "timestamp": payload.get("timestamp", ""),
            "schema": payload.get("schema", "public")
        }
        
        # Process the database change
        await realtime_service.handle_database_change(change_event)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Webhook processed successfully",
            "processed_at": realtime_service.last_sync_time.isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Supabase webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


@router.post("/generic")
async def generic_webhook_handler(
    request: Request,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle generic webhook for testing and development.
    
    Args:
        request: FastAPI request object
        payload: Webhook payload
        
    Returns:
        Acknowledgment response
    """
    try:
        logger.info(f"Received generic webhook: {payload}")
        
        # Create standardized change event
        change_event = {
            "type": payload.get("event_type", "UPDATE"),
            "table": payload.get("table", "customers"),
            "record": payload.get("data", {}),
            "old_record": payload.get("old_data", {}),
            "timestamp": payload.get("timestamp", ""),
            "source": "generic_webhook"
        }
        
        # Process the change
        await realtime_service.handle_database_change(change_event)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Generic webhook processed successfully",
            "change_event": change_event
        })
        
    except Exception as e:
        logger.error(f"Error processing generic webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


@router.get("/test")
async def test_webhook_system() -> Dict[str, Any]:
    """
    Test the webhook system by simulating a database change.
    
    Returns:
        Test results
    """
    try:
        logger.info("Testing webhook system")
        
        # Create test change event
        test_change = {
            "type": "UPDATE",
            "table": "customers",
            "record": {
                "customer_id": "TEST_WEBHOOK_001",
                "churn": 1,
                "monthly_charges": 85.50
            },
            "old_record": {
                "customer_id": "TEST_WEBHOOK_001",
                "churn": 0,
                "monthly_charges": 75.50
            },
            "timestamp": realtime_service.last_sync_time.isoformat(),
            "source": "webhook_test"
        }
        
        # Process the test change
        await realtime_service.handle_database_change(test_change)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Webhook system test completed",
            "test_change": test_change,
            "active_connections": len(realtime_service.connection_manager.active_connections)
        })
        
    except Exception as e:
        logger.error(f"Error testing webhook system: {e}")
        raise HTTPException(status_code=500, detail=f"Webhook test failed: {str(e)}")