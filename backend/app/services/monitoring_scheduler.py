"""
Background scheduler for automatic model monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import threading
import time

from app.core.logging import get_logger
from app.services.model_monitoring import get_model_monitor

logger = get_logger(__name__)


class MonitoringScheduler:
    """Background scheduler for automatic model monitoring."""
    
    def __init__(self, 
                 monitoring_interval_hours: int = 24,
                 evaluation_interval_hours: int = 6):
        self.monitoring_interval_hours = monitoring_interval_hours
        self.evaluation_interval_hours = evaluation_interval_hours
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_full_monitoring = None
        self._last_evaluation = None
        
    def start(self) -> None:
        """Start the monitoring scheduler."""
        if self._running:
            logger.warning("Monitoring scheduler is already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        
        logger.info(f"Monitoring scheduler started: "
                   f"full monitoring every {self.monitoring_interval_hours}h, "
                   f"evaluation every {self.evaluation_interval_hours}h")
    
    def stop(self) -> None:
        """Stop the monitoring scheduler."""
        if not self._running:
            return
            
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            
        logger.info("Monitoring scheduler stopped")
    
    def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check if it's time for evaluation
                if self._should_run_evaluation(current_time):
                    asyncio.run(self._run_evaluation())
                    self._last_evaluation = current_time
                
                # Check if it's time for full monitoring cycle
                elif self._should_run_full_monitoring(current_time):
                    asyncio.run(self._run_full_monitoring())
                    self._last_full_monitoring = current_time
                
                # Sleep for 30 minutes before next check
                time.sleep(30 * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring scheduler: {e}")
                # Sleep for 5 minutes before retrying
                time.sleep(5 * 60)
    
    def _should_run_evaluation(self, current_time: datetime) -> bool:
        """Check if evaluation should run."""
        if self._last_evaluation is None:
            return True
            
        time_since_last = current_time - self._last_evaluation
        return time_since_last >= timedelta(hours=self.evaluation_interval_hours)
    
    def _should_run_full_monitoring(self, current_time: datetime) -> bool:
        """Check if full monitoring cycle should run."""
        if self._last_full_monitoring is None:
            return True
            
        time_since_last = current_time - self._last_full_monitoring
        return time_since_last >= timedelta(hours=self.monitoring_interval_hours)
    
    async def _run_evaluation(self) -> None:
        """Run model evaluation only."""
        try:
            logger.info("Running scheduled model evaluation")
            
            monitor = get_model_monitor()
            metrics = await monitor.evaluate_model_performance()
            
            logger.info(f"Scheduled evaluation completed: "
                       f"accuracy={metrics.accuracy:.3f}, "
                       f"f1_score={metrics.f1_score:.3f}")
                       
        except Exception as e:
            logger.error(f"Scheduled evaluation failed: {e}")
    
    async def _run_full_monitoring(self) -> None:
        """Run full monitoring cycle."""
        try:
            logger.info("Running scheduled monitoring cycle")
            
            monitor = get_model_monitor()
            results = await monitor.run_monitoring_cycle()
            
            status = results.get('status', 'unknown')
            model_status = results.get('model_status', 'unknown')
            
            if results.get('should_retrain', False):
                trigger_reason = results.get('trigger_reason', 'unknown')
                logger.info(f"Scheduled monitoring triggered retraining: {trigger_reason}")
            
            logger.info(f"Scheduled monitoring completed: "
                       f"status={status}, model_status={model_status}")
                       
        except Exception as e:
            logger.error(f"Scheduled monitoring failed: {e}")
    
    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "monitoring_interval_hours": self.monitoring_interval_hours,
            "evaluation_interval_hours": self.evaluation_interval_hours,
            "last_full_monitoring": self._last_full_monitoring.isoformat() if self._last_full_monitoring else None,
            "last_evaluation": self._last_evaluation.isoformat() if self._last_evaluation else None,
            "next_evaluation": (
                self._last_evaluation + timedelta(hours=self.evaluation_interval_hours)
            ).isoformat() if self._last_evaluation else "immediately",
            "next_full_monitoring": (
                self._last_full_monitoring + timedelta(hours=self.monitoring_interval_hours)
            ).isoformat() if self._last_full_monitoring else "immediately"
        }


# Global scheduler instance
monitoring_scheduler = None

def get_monitoring_scheduler() -> MonitoringScheduler:
    """Get or create monitoring scheduler instance."""
    global monitoring_scheduler
    if monitoring_scheduler is None:
        monitoring_scheduler = MonitoringScheduler()
    return monitoring_scheduler

def start_monitoring_scheduler() -> None:
    """Start the global monitoring scheduler."""
    scheduler = get_monitoring_scheduler()
    scheduler.start()

def stop_monitoring_scheduler() -> None:
    """Stop the global monitoring scheduler."""
    global monitoring_scheduler
    if monitoring_scheduler:
        monitoring_scheduler.stop()
        monitoring_scheduler = None