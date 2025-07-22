"""
Real-time WebSocket endpoints.
"""

import json
import uuid
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from app.services.realtime import realtime_service
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await realtime_service.connection_manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": realtime_service.last_sync_time.isoformat(),
            "message": "Connected to real-time updates"
        }
        await realtime_service.connection_manager.send_personal_message(welcome_message, client_id)
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(message, client_id)
            
    except WebSocketDisconnect:
        realtime_service.connection_manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        realtime_service.connection_manager.disconnect(client_id)


async def handle_websocket_message(message: Dict[str, Any], client_id: str):
    """
    Handle incoming WebSocket messages from clients.
    
    Args:
        message: Message data from client
        client_id: Client identifier
    """
    try:
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Subscribe to event types
            event_types = message.get("event_types", [])
            realtime_service.connection_manager.subscribe(client_id, event_types)
            
            response = {
                "type": "subscription_confirmed",
                "event_types": event_types,
                "client_id": client_id
            }
            await realtime_service.connection_manager.send_personal_message(response, client_id)
            
        elif message_type == "unsubscribe":
            # Unsubscribe from event types
            event_types = message.get("event_types", [])
            realtime_service.connection_manager.unsubscribe(client_id, event_types)
            
            response = {
                "type": "unsubscription_confirmed",
                "event_types": event_types,
                "client_id": client_id
            }
            await realtime_service.connection_manager.send_personal_message(response, client_id)
            
        elif message_type == "ping":
            # Respond to ping
            response = {
                "type": "pong",
                "timestamp": realtime_service.last_sync_time.isoformat()
            }
            await realtime_service.connection_manager.send_personal_message(response, client_id)
            
        elif message_type == "get_status":
            # Send sync status
            status = await realtime_service.get_sync_status()
            response = {
                "type": "sync_status",
                **status
            }
            await realtime_service.connection_manager.send_personal_message(response, client_id)
            
        else:
            # Unknown message type
            response = {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }
            await realtime_service.connection_manager.send_personal_message(response, client_id)
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message from {client_id}: {e}")
        error_response = {
            "type": "error",
            "message": f"Error processing message: {str(e)}"
        }
        await realtime_service.connection_manager.send_personal_message(error_response, client_id)


@router.get("/sync-status")
async def get_sync_status() -> Dict[str, Any]:
    """
    Get current synchronization status.
    
    Returns:
        Dictionary containing sync status information
    """
    try:
        logger.info("Getting sync status")
        status = await realtime_service.get_sync_status()
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sync status: {str(e)}")


@router.post("/manual-sync")
async def trigger_manual_sync() -> Dict[str, Any]:
    """
    Trigger manual data synchronization.
    
    Returns:
        Status of the manual sync operation
    """
    try:
        logger.info("Manual sync triggered via API")
        result = await realtime_service.manual_sync()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error triggering manual sync: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger manual sync: {str(e)}")


@router.post("/simulate-change")
async def simulate_database_change(change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a database change for testing purposes.
    
    Args:
        change_data: Simulated change event data
        
    Returns:
        Confirmation of simulated change
    """
    try:
        logger.info(f"Simulating database change: {change_data}")
        
        # Add default values if not provided
        simulated_change = {
            "type": change_data.get("type", "UPDATE"),
            "table": change_data.get("table", "customers"),
            "record": change_data.get("record", {"customer_id": "SIM_001", "churn": 1}),
            "old_record": change_data.get("old_record", {}),
            "timestamp": realtime_service.last_sync_time.isoformat()
        }
        
        # Process the simulated change
        await realtime_service.handle_database_change(simulated_change)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Database change simulated successfully",
            "simulated_change": simulated_change
        })
        
    except Exception as e:
        logger.error(f"Error simulating database change: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to simulate change: {str(e)}")


@router.get("/connections")
async def get_active_connections() -> Dict[str, Any]:
    """
    Get information about active WebSocket connections.
    
    Returns:
        Information about connected clients
    """
    try:
        connections_info = {
            "active_connections": len(realtime_service.connection_manager.active_connections),
            "client_ids": list(realtime_service.connection_manager.active_connections.keys()),
            "subscriptions": {
                client_id: list(subs) 
                for client_id, subs in realtime_service.connection_manager.subscriptions.items()
            }
        }
        
        return JSONResponse(content=connections_info)
        
    except Exception as e:
        logger.error(f"Error getting connection info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get connection info: {str(e)}")