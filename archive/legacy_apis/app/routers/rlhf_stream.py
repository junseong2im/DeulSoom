"""
RLHF Stream Router
Server-Sent Events (SSE) for real-time RLHF training status streaming
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime

from app.auth import get_current_user_id, get_current_user_optional

router = APIRouter(prefix="/stream", tags=["RLHF Stream"])
logger = logging.getLogger(__name__)


# ============================================================================
# SSE Helper Functions
# ============================================================================

def format_sse(data: dict, event: Optional[str] = None) -> str:
    """
    Format data as Server-Sent Events message

    Args:
        data: Data to send
        event: Optional event type

    Returns:
        SSE formatted string
    """
    message = ""

    if event:
        message += f"event: {event}\n"

    message += f"data: {json.dumps(data)}\n\n"

    return message


async def keep_alive_generator(interval: int = 30) -> AsyncGenerator[str, None]:
    """
    Generate keep-alive messages to prevent connection timeout

    Args:
        interval: Interval in seconds

    Yields:
        SSE keep-alive messages
    """
    while True:
        await asyncio.sleep(interval)
        yield format_sse({"type": "keep-alive", "timestamp": datetime.utcnow().isoformat()})


# ============================================================================
# RLHF Training Status Stream
# ============================================================================

@router.get("/rlhf/training/{experiment_id}")
async def stream_rlhf_training(
    experiment_id: str,
    user_id: Optional[str] = Depends(get_current_user_optional)
):
    """
    Stream RLHF training status in real-time using SSE

    - Sends training metrics every iteration
    - Includes loss, reward, entropy
    - Automatically closes when training completes

    **SSE Event Types:**
    - `training_started`: Training session initialized
    - `iteration`: Training iteration completed
    - `metrics`: Real-time metrics update
    - `completed`: Training finished
    - `error`: Training error occurred
    - `keep-alive`: Connection keep-alive ping

    **Usage (JavaScript):**
    ```javascript
    const eventSource = new EventSource('/stream/rlhf/training/exp_123');

    eventSource.addEventListener('metrics', (e) => {
        const data = JSON.parse(e.data);
        console.log('Metrics:', data);
    });

    eventSource.addEventListener('completed', (e) => {
        eventSource.close();
    });
    ```
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for RLHF training"""

        try:
            # Send initialization event
            yield format_sse({
                "type": "training_started",
                "experiment_id": experiment_id,
                "user_id": user_id or "anonymous",
                "timestamp": datetime.utcnow().isoformat()
            }, event="training_started")

            logger.info(f"Started SSE stream for experiment {experiment_id}")

            # Simulate training iterations (in production, this would listen to actual training events)
            # You would integrate this with your actual RLHF training loop

            for iteration in range(1, 11):  # Simulate 10 iterations
                await asyncio.sleep(2)  # Simulate training time

                # Simulate metrics (replace with actual metrics from training)
                metrics = {
                    "iteration": iteration,
                    "loss": 0.5 - (iteration * 0.03),
                    "reward": 0.3 + (iteration * 0.05),
                    "entropy": 0.8 - (iteration * 0.02),
                    "policy_loss": 0.4 - (iteration * 0.02),
                    "value_loss": 0.3 - (iteration * 0.01),
                    "learning_rate": 0.0003,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Send iteration event
                yield format_sse({
                    "type": "iteration",
                    "experiment_id": experiment_id,
                    **metrics
                }, event="iteration")

                # Send metrics event (same data, different event type for flexibility)
                yield format_sse({
                    "type": "metrics",
                    **metrics
                }, event="metrics")

                logger.debug(f"Sent metrics for iteration {iteration}")

            # Training completed
            yield format_sse({
                "type": "completed",
                "experiment_id": experiment_id,
                "total_iterations": 10,
                "final_loss": metrics["loss"],
                "final_reward": metrics["reward"],
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }, event="completed")

            logger.info(f"Completed SSE stream for experiment {experiment_id}")

        except asyncio.CancelledError:
            # Client disconnected
            logger.info(f"Client disconnected from SSE stream {experiment_id}")
            yield format_sse({
                "type": "disconnected",
                "experiment_id": experiment_id,
                "timestamp": datetime.utcnow().isoformat()
            }, event="disconnected")

        except Exception as e:
            # Error occurred
            logger.error(f"Error in SSE stream {experiment_id}: {e}")
            yield format_sse({
                "type": "error",
                "experiment_id": experiment_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }, event="error")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# ============================================================================
# General Purpose Progress Stream
# ============================================================================

@router.get("/progress/{task_id}")
async def stream_task_progress(
    task_id: str,
    user_id: Optional[str] = Depends(get_current_user_optional)
):
    """
    Stream general task progress (DNA creation, optimization, etc.)

    - Generic progress streaming endpoint
    - Supports any long-running task
    - Sends progress percentage and status updates
    """

    async def progress_generator() -> AsyncGenerator[str, None]:
        """Generate progress events"""

        try:
            # Send start event
            yield format_sse({
                "type": "started",
                "task_id": task_id,
                "status": "processing",
                "progress": 0,
                "timestamp": datetime.utcnow().isoformat()
            }, event="started")

            # Simulate progress updates
            for progress in range(0, 101, 10):
                await asyncio.sleep(1)

                yield format_sse({
                    "type": "progress",
                    "task_id": task_id,
                    "progress": progress,
                    "status": "processing" if progress < 100 else "completed",
                    "timestamp": datetime.utcnow().isoformat()
                }, event="progress")

            # Send completion event
            yield format_sse({
                "type": "completed",
                "task_id": task_id,
                "progress": 100,
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }, event="completed")

        except asyncio.CancelledError:
            yield format_sse({
                "type": "cancelled",
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat()
            }, event="cancelled")

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ============================================================================
# Real-time Experiment Updates
# ============================================================================

@router.get("/experiments/{experiment_id}/live")
async def stream_experiment_updates(
    experiment_id: str,
    user_id: Optional[str] = Depends(get_current_user_optional)
):
    """
    Stream live experiment updates

    - User feedback events
    - Iteration completion
    - Policy updates
    - Optimization results
    """

    async def experiment_generator() -> AsyncGenerator[str, None]:
        """Generate experiment update events"""

        try:
            yield format_sse({
                "type": "connected",
                "experiment_id": experiment_id,
                "timestamp": datetime.utcnow().isoformat()
            }, event="connected")

            # Keep connection alive and wait for actual events
            # In production, this would listen to a message queue or event bus

            keep_alive_task = asyncio.create_task(
                _keep_alive_loop()
            )

            # Wait for events (placeholder - integrate with actual event system)
            await asyncio.sleep(300)  # 5 minutes max

            keep_alive_task.cancel()

        except asyncio.CancelledError:
            yield format_sse({
                "type": "disconnected",
                "experiment_id": experiment_id,
                "timestamp": datetime.utcnow().isoformat()
            }, event="disconnected")

    return StreamingResponse(
        experiment_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


async def _keep_alive_loop():
    """Keep connection alive with periodic pings"""
    while True:
        await asyncio.sleep(30)


# ============================================================================
# Export
# ============================================================================

__all__ = ['router']
