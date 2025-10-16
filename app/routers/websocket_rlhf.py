"""
WebSocket RLHF Router
WebSocket endpoints for bi-directional real-time RLHF communication
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, Optional, Set
import asyncio
import json
import logging
from datetime import datetime

router = APIRouter(prefix="/ws", tags=["WebSocket"])
logger = logging.getLogger(__name__)


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        # Map of experiment_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: str):
        """Connect a WebSocket to an experiment"""
        await websocket.accept()

        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()

        self.active_connections[experiment_id].add(websocket)
        logger.info(f"WebSocket connected to experiment {experiment_id}")

    def disconnect(self, websocket: WebSocket, experiment_id: str):
        """Disconnect a WebSocket from an experiment"""
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)

            # Clean up empty experiment groups
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]

        logger.info(f"WebSocket disconnected from experiment {experiment_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast_to_experiment(self, experiment_id: str, message: dict):
        """Broadcast message to all connections in an experiment"""
        if experiment_id not in self.active_connections:
            return

        disconnected = set()

        for connection in self.active_connections[experiment_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws, experiment_id)


# Global connection manager
manager = ConnectionManager()


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/rlhf/training/{experiment_id}")
async def websocket_rlhf_training(
    websocket: WebSocket,
    experiment_id: str,
    token: Optional[str] = Query(None, description="JWT token for authentication")
):
    """
    WebSocket endpoint for real-time RLHF training updates

    **Features:**
    - Bi-directional communication
    - Real-time training metrics
    - User can send commands (pause, resume, stop)
    - Multiple clients can connect to same experiment

    **Client Messages:**
    - `{"type": "subscribe"}`: Subscribe to updates
    - `{"type": "pause"}`: Pause training
    - `{"type": "resume"}`: Resume training
    - `{"type": "stop"}`: Stop training

    **Server Messages:**
    - `{"type": "connected"}`: Connection established
    - `{"type": "metrics", ...}`: Training metrics update
    - `{"type": "status", ...}`: Training status change
    - `{"type": "completed"}`: Training finished
    - `{"type": "error", ...}`: Error occurred

    **Usage (JavaScript):**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/rlhf/training/exp_123?token=YOUR_JWT');

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data);
    };

    // Send command
    ws.send(JSON.stringify({ type: 'pause' }));
    ```
    """

    # TODO: Validate JWT token if provided
    # if token:
    #     user_id = verify_jwt_token(token)
    # else:
    #     user_id = "anonymous"

    await manager.connect(websocket, experiment_id)

    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

        # Start training simulation (replace with actual training loop)
        training_task = asyncio.create_task(
            _simulate_training(experiment_id, websocket)
        )

        # Listen for client messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()

                logger.info(f"Received from client: {data}")

                # Handle client commands
                message_type = data.get("type")

                if message_type == "subscribe":
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "experiment_id": experiment_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)

                elif message_type == "pause":
                    # TODO: Implement pause logic
                    await manager.send_personal_message({
                        "type": "status",
                        "status": "paused",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)

                elif message_type == "resume":
                    # TODO: Implement resume logic
                    await manager.send_personal_message({
                        "type": "status",
                        "status": "resumed",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)

                elif message_type == "stop":
                    # Stop training
                    training_task.cancel()
                    await manager.send_personal_message({
                        "type": "status",
                        "status": "stopped",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)
                    break

                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    }, websocket)

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for experiment {experiment_id}")
                break

            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from experiment {experiment_id}")

    finally:
        manager.disconnect(websocket, experiment_id)


async def _simulate_training(experiment_id: str, websocket: WebSocket):
    """Simulate training and send metrics"""

    try:
        for iteration in range(1, 11):
            await asyncio.sleep(2)

            # Simulate metrics
            metrics = {
                "type": "metrics",
                "experiment_id": experiment_id,
                "iteration": iteration,
                "loss": 0.5 - (iteration * 0.03),
                "reward": 0.3 + (iteration * 0.05),
                "entropy": 0.8 - (iteration * 0.02),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Send to specific websocket
            await manager.send_personal_message(metrics, websocket)

            # Also broadcast to all connected clients
            await manager.broadcast_to_experiment(experiment_id, {
                "type": "broadcast",
                "message": f"Iteration {iteration} completed",
                "timestamp": datetime.utcnow().isoformat()
            })

        # Training completed
        await manager.send_personal_message({
            "type": "completed",
            "experiment_id": experiment_id,
            "total_iterations": 10,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

    except asyncio.CancelledError:
        logger.info(f"Training simulation cancelled for {experiment_id}")


@router.websocket("/experiments/{experiment_id}/live")
async def websocket_experiment_live(
    websocket: WebSocket,
    experiment_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket for general experiment live updates

    - Feedback submissions
    - Option selections
    - Real-time collaboration
    """

    await manager.connect(websocket, experiment_id)

    try:
        await manager.send_personal_message({
            "type": "connected",
            "experiment_id": experiment_id,
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)

        # Listen for messages
        while True:
            try:
                data = await websocket.receive_json()

                # Broadcast to all connected clients
                await manager.broadcast_to_experiment(experiment_id, {
                    "type": "update",
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                })

            except WebSocketDisconnect:
                break

    finally:
        manager.disconnect(websocket, experiment_id)


# ============================================================================
# Export
# ============================================================================

__all__ = ['router', 'manager']
