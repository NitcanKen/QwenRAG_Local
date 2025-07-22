"""
Main API router for version 1.
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import customers, analytics, ml, rag, realtime, webhooks

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(customers.router, prefix="/customers", tags=["customers"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(ml.router, prefix="/ml", tags=["machine-learning"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(realtime.router, prefix="/realtime", tags=["realtime"])
api_router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])