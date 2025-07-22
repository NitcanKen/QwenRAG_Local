"""
Real-time data synchronization service.
"""

import asyncio
import json
from typing import Dict, List, Set, Any, Optional
from datetime import datetime, timedelta
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum

from app.core.logging import get_logger
from app.services.analytics import analytics_service

logger = get_logger(__name__)


class ChangeEventType(str, Enum):
    """Types of database change events."""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    REFRESH = "REFRESH"


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Store active connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Track subscriptions per connection
        self.subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """
        Disconnect a WebSocket client.
        
        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """
        Send a message to a specific client.
        
        Args:
            message: Message to send
            client_id: Target client identifier
        """
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict, event_type: str = None):
        """
        Broadcast a message to all connected clients or subscribed clients.
        
        Args:
            message: Message to broadcast
            event_type: Optional event type for filtering subscriptions
        """
        if not self.active_connections:
            return
        
        # If event_type is specified, only send to subscribed clients
        target_clients = []
        if event_type:
            for client_id, subscriptions in self.subscriptions.items():
                if event_type in subscriptions or "all" in subscriptions:
                    target_clients.append(client_id)
        else:
            target_clients = list(self.active_connections.keys())
        
        # Send message to target clients
        disconnected_clients = []
        for client_id in target_clients:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
        
        if target_clients:
            logger.info(f"Broadcast message to {len(target_clients)} clients")
    
    def subscribe(self, client_id: str, event_types: List[str]):
        """
        Subscribe a client to specific event types.
        
        Args:
            client_id: Client identifier
            event_types: List of event types to subscribe to
        """
        if client_id in self.subscriptions:
            self.subscriptions[client_id].update(event_types)
            logger.info(f"Client {client_id} subscribed to: {event_types}")
    
    def unsubscribe(self, client_id: str, event_types: List[str]):
        """
        Unsubscribe a client from specific event types.
        
        Args:
            client_id: Client identifier
            event_types: List of event types to unsubscribe from
        """
        if client_id in self.subscriptions:
            self.subscriptions[client_id] -= set(event_types)
            logger.info(f"Client {client_id} unsubscribed from: {event_types}")


class RealtimeDataService:
    """Service for managing real-time data synchronization."""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.last_sync_time = datetime.now()
        self.sync_interval = timedelta(minutes=5)  # 5-minute sync interval
        self.is_syncing = False
    
    async def handle_database_change(self, change_event: Dict[str, Any]):
        """
        Handle database change events from Supabase or other sources.
        
        Args:
            change_event: Database change event data
        """
        try:
            logger.info(f"Processing database change: {change_event}")
            
            # Extract change information
            event_type = change_event.get("type", "UNKNOWN")
            table = change_event.get("table", "unknown")
            record = change_event.get("record", {})
            old_record = change_event.get("old_record", {})
            
            # Create notification message
            message = {
                "type": "database_change",
                "event_type": event_type,
                "table": table,
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "record": record,
                    "old_record": old_record if old_record else None
                }
            }
            
            # Broadcast to subscribed clients
            await self.connection_manager.broadcast(message, "database_changes")
            
            # If it's a customer table change, trigger analytics refresh
            if table == "customers":
                await self.trigger_analytics_refresh()
            
        except Exception as e:
            logger.error(f"Error handling database change: {e}")
    
    async def trigger_analytics_refresh(self):
        """Trigger analytics data refresh and notify clients."""
        try:
            logger.info("Triggering analytics refresh due to data changes")
            
            # Clear analytics cache (will force fresh calculations)
            await self.clear_analytics_cache()
            
            # Notify clients that analytics need refresh
            message = {
                "type": "analytics_refresh",
                "timestamp": datetime.now().isoformat(),
                "message": "Analytics data has been updated due to database changes"
            }
            
            await self.connection_manager.broadcast(message, "analytics_updates")
            
        except Exception as e:
            logger.error(f"Error triggering analytics refresh: {e}")
    
    async def clear_analytics_cache(self):
        """Clear analytics cache to force fresh data on next request."""
        try:
            # Clear in-memory cache
            from app.services.analytics import _cache
            _cache.clear()
            
            # Clear Redis cache if available
            try:
                import redis
                from app.core.config import settings
                redis_client = redis.Redis.from_url(settings.REDIS_URL)
                
                # Get all analytics cache keys and delete them
                pattern = "*"  # This is basic - in production you'd want more specific patterns
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} Redis cache entries")
            except Exception as e:
                logger.debug(f"Redis cache clear failed: {e}")
            
            logger.info("Analytics cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing analytics cache: {e}")
    
    async def periodic_sync(self):
        """Perform periodic data synchronization."""
        try:
            if self.is_syncing:
                logger.debug("Sync already in progress, skipping")
                return
            
            now = datetime.now()
            if now - self.last_sync_time < self.sync_interval:
                logger.debug("Sync interval not reached, skipping")
                return
            
            self.is_syncing = True
            logger.info("Starting periodic data synchronization")
            
            # Perform data aggregation and analysis
            await self.refresh_analytics_data()
            
            # Update last sync time
            self.last_sync_time = now
            
            # Notify clients about sync completion
            message = {
                "type": "sync_complete",
                "timestamp": now.isoformat(),
                "message": "Periodic data synchronization completed"
            }
            
            await self.connection_manager.broadcast(message, "sync_updates")
            
        except Exception as e:
            logger.error(f"Error during periodic sync: {e}")
        finally:
            self.is_syncing = False
    
    async def refresh_analytics_data(self):
        """Refresh analytics data in background."""
        try:
            logger.info("Refreshing analytics data")
            
            # Pre-compute analytics to warm up cache
            analytics_functions = [
                analytics_service.get_overall_churn_metrics,
                analytics_service.get_churn_rate_by_tenure,
                analytics_service.get_churn_rate_by_contract,
                analytics_service.get_demographic_analysis,
                analytics_service.get_service_impact_analysis,
                analytics_service.get_financial_metrics
            ]
            
            # Execute all analytics functions to warm cache
            tasks = [func() for func in analytics_functions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            success_count = sum(1 for result in results if not isinstance(result, Exception))
            error_count = len(results) - success_count
            
            logger.info(f"Analytics refresh completed: {success_count} success, {error_count} errors")
            
            if error_count > 0:
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Analytics function {i} failed: {result}")
            
        except Exception as e:
            logger.error(f"Error refreshing analytics data: {e}")
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current synchronization status.
        
        Returns:
            Dictionary containing sync status information
        """
        return {
            "last_sync_time": self.last_sync_time.isoformat(),
            "is_syncing": self.is_syncing,
            "sync_interval_minutes": self.sync_interval.total_seconds() / 60,
            "connected_clients": len(self.connection_manager.active_connections),
            "next_sync_in_minutes": max(0, (
                self.last_sync_time + self.sync_interval - datetime.now()
            ).total_seconds() / 60)
        }
    
    async def manual_sync(self) -> Dict[str, Any]:
        """
        Trigger manual synchronization.
        
        Returns:
            Status of the manual sync operation
        """
        try:
            logger.info("Manual sync triggered")
            await self.refresh_analytics_data()
            
            # Notify clients
            message = {
                "type": "manual_sync_complete",
                "timestamp": datetime.now().isoformat(),
                "message": "Manual synchronization completed"
            }
            
            await self.connection_manager.broadcast(message, "sync_updates")
            
            return {
                "status": "success",
                "message": "Manual sync completed successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during manual sync: {e}")
            return {
                "status": "error",
                "message": f"Manual sync failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


# Global instance
realtime_service = RealtimeDataService()