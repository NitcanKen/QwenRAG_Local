"""
Background task service for periodic data synchronization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from app.services.realtime import realtime_service
from app.core.logging import get_logger

logger = get_logger(__name__)


class BackgroundTaskService:
    """Service for managing background tasks."""
    
    def __init__(self):
        self.sync_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start_background_tasks(self):
        """Start all background tasks."""
        if self.is_running:
            logger.warning("Background tasks already running")
            return
        
        self.is_running = True
        logger.info("Starting background tasks")
        
        # Start periodic sync task
        self.sync_task = asyncio.create_task(self.periodic_sync_loop())
        
        logger.info("Background tasks started successfully")
    
    async def stop_background_tasks(self):
        """Stop all background tasks."""
        if not self.is_running:
            logger.warning("Background tasks not running")
            return
        
        logger.info("Stopping background tasks")
        self.is_running = False
        
        # Cancel sync task
        if self.sync_task and not self.sync_task.done():
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background tasks stopped")
    
    async def periodic_sync_loop(self):
        """Main loop for periodic data synchronization."""
        logger.info("Starting periodic sync loop")
        
        try:
            while self.is_running:
                try:
                    # Perform periodic sync
                    await realtime_service.periodic_sync()
                    
                    # Wait for next cycle (check every 30 seconds)
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Error in periodic sync loop: {e}")
                    # Wait a bit before retrying
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Periodic sync loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in periodic sync loop: {e}")
        finally:
            logger.info("Periodic sync loop ended")


# Global instance
background_task_service = BackgroundTaskService()