"""
WebSocket connection manager for GenomeVault API Gateway.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from genomevault.api.gateway.models.websockets import (
    ConnectionInfo,
    MessageType,
    WebSocketMessage,
    WebSocketResponse,
)
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and message routing.
    
    Features:
    - Connection lifecycle management
    - Message broadcasting
    - User-based connection grouping
    - Connection health monitoring
    - Automatic cleanup of stale connections
    """
    
    def __init__(self):
        """Initialize connection manager."""
        # Active connections by connection ID
        self.connections: Dict[str, WebSocket] = {}
        
        # Connection metadata
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # User to connections mapping
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Connection statistics
        self.connection_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"messages_sent": 0, "messages_received": 0}
        )
        
        # Ping/pong tracking for connection health
        self.pending_pings: Dict[str, float] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: Optional[str] = None,
        user_id: Optional[str] = None,
        client_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            connection_id: Optional connection ID (generated if not provided)
            user_id: Optional authenticated user ID
            client_info: Optional client information
            
        Returns:
            Connection identifier
        """
        # Generate connection ID if not provided
        if not connection_id:
            connection_id = f"conn_{int(time.time() * 1000000)}"
        
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Register connection
        self.connections[connection_id] = websocket
        
        # Create connection info
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            client_info=client_info or {},
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            active_subscriptions=[],
            subscription_count=0,
            messages_sent=0,
            messages_received=0
        )
        
        self.connection_info[connection_id] = connection_info
        
        # Map user to connection if user is authenticated
        if user_id:
            self.user_connections[user_id].add(connection_id)
        
        # Initialize statistics
        self.connection_stats[connection_id] = {
            "messages_sent": 0,
            "messages_received": 0
        }
        
        # Start background tasks if this is the first connection
        if len(self.connections) == 1:
            await self._start_background_tasks()
        
        logger.info(
            f"WebSocket connection established: {connection_id}",
            extra={
                "connection_id": connection_id,
                "user_id": user_id,
                "total_connections": len(self.connections)
            }
        )
        
        # Send connection confirmation
        await self.send_message(connection_id, WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.CONNECT,
            timestamp=datetime.utcnow(),
            data={
                "connection_id": connection_id,
                "status": "connected",
                "server_time": datetime.utcnow().isoformat()
            }
        ))
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Disconnect and clean up a WebSocket connection.
        
        Args:
            connection_id: Connection identifier
        """
        if connection_id not in self.connections:
            return
        
        # Get connection info
        conn_info = self.connection_info.get(connection_id)
        
        # Remove from user mapping
        if conn_info and conn_info.user_id:
            self.user_connections[conn_info.user_id].discard(connection_id)
            if not self.user_connections[conn_info.user_id]:
                del self.user_connections[conn_info.user_id]
        
        # Clean up subscriptions (handled by subscription manager)
        from genomevault.api.gateway.websockets.subscription_manager import subscription_manager
        await subscription_manager.cleanup_connection_subscriptions(connection_id)
        
        # Remove connection
        websocket = self.connections.pop(connection_id, None)
        self.connection_info.pop(connection_id, None)
        self.connection_stats.pop(connection_id, None)
        self.pending_pings.pop(connection_id, None)
        
        # Close WebSocket if still open
        if websocket:
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket {connection_id}: {e}")
        
        logger.info(
            f"WebSocket connection disconnected: {connection_id}",
            extra={
                "connection_id": connection_id,
                "user_id": conn_info.user_id if conn_info else None,
                "total_connections": len(self.connections),
                "connection_duration_seconds": (
                    (datetime.utcnow() - conn_info.connected_at).total_seconds()
                    if conn_info else 0
                )
            }
        )
        
        # Stop background tasks if no connections remain
        if not self.connections:
            await self._stop_background_tasks()
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Connection identifier
            message: Message to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        websocket = self.connections.get(connection_id)
        if not websocket:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return False
        
        try:
            # Convert message to JSON
            message_json = message.json()
            
            # Send message
            await websocket.send_text(message_json)
            
            # Update statistics
            self.connection_stats[connection_id]["messages_sent"] += 1
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id].messages_sent += 1
                self.connection_info[connection_id].last_activity = datetime.utcnow()
            
            logger.debug(
                f"Sent WebSocket message: {message.message_type}",
                extra={
                    "connection_id": connection_id,
                    "message_id": message.message_id,
                    "message_type": message.message_type
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to send WebSocket message to {connection_id}: {e}",
                extra={
                    "connection_id": connection_id,
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
            
            # Connection might be broken, schedule for cleanup
            asyncio.create_task(self.disconnect(connection_id))
            
            return False
    
    async def send_response(self, connection_id: str, response: WebSocketResponse) -> bool:
        """
        Send a response message to a specific connection.
        
        Args:
            connection_id: Connection identifier
            response: Response to send
            
        Returns:
            True if response was sent successfully, False otherwise
        """
        # Convert response to WebSocketMessage format
        message = WebSocketMessage(
            message_id=response.message_id,
            message_type=response.message_type,
            timestamp=response.timestamp,
            data={
                "request_message_id": response.request_message_id,
                "success": response.success,
                "data": response.data,
                "error": response.error,
                "error_details": response.error_details
            }
        )
        
        return await self.send_message(connection_id, message)
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all connections for a specific user.
        
        Args:
            user_id: User identifier
            message: Message to broadcast
            
        Returns:
            Number of connections the message was sent to
        """
        connection_ids = self.user_connections.get(user_id, set())
        sent_count = 0
        
        for connection_id in connection_ids.copy():  # Copy to avoid modification during iteration
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: WebSocketMessage) -> int:
        """
        Broadcast a message to all active connections.
        
        Args:
            message: Message to broadcast
            
        Returns:
            Number of connections the message was sent to
        """
        sent_count = 0
        
        for connection_id in list(self.connections.keys()):  # Create list to avoid modification during iteration
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def receive_message(self, connection_id: str, websocket: WebSocket) -> Optional[WebSocketMessage]:
        """
        Receive and parse a message from a WebSocket connection.
        
        Args:
            connection_id: Connection identifier
            websocket: WebSocket connection
            
        Returns:
            Parsed WebSocket message or None if connection closed
        """
        try:
            # Receive message
            data = await websocket.receive_text()
            
            # Parse JSON
            message_dict = json.loads(data)
            
            # Create WebSocket message object
            message = WebSocketMessage(**message_dict)
            
            # Update statistics
            self.connection_stats[connection_id]["messages_received"] += 1
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id].messages_received += 1
                self.connection_info[connection_id].last_activity = datetime.utcnow()
            
            logger.debug(
                f"Received WebSocket message: {message.message_type}",
                extra={
                    "connection_id": connection_id,
                    "message_id": message.message_id,
                    "message_type": message.message_type
                }
            )
            
            return message
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
            await self.disconnect(connection_id)
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON received from {connection_id}: {e}",
                extra={"connection_id": connection_id}
            )
            
            # Send error response
            error_response = WebSocketResponse(
                message_id=str(uuid.uuid4()),
                request_message_id=None,
                message_type=MessageType.ERROR,
                timestamp=datetime.utcnow(),
                success=False,
                error="Invalid JSON format"
            )
            await self.send_response(connection_id, error_response)
            
            return None
            
        except Exception as e:
            logger.error(
                f"Error receiving WebSocket message from {connection_id}: {e}",
                extra={"connection_id": connection_id, "error": str(e)}
            )
            return None
    
    async def ping_connection(self, connection_id: str) -> bool:
        """
        Send a ping to test connection health.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if ping was sent successfully
        """
        ping_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PING,
            timestamp=datetime.utcnow(),
            data={"timestamp": time.time()}
        )
        
        success = await self.send_message(connection_id, ping_message)
        
        if success:
            self.pending_pings[connection_id] = time.time()
        
        return success
    
    async def handle_pong(self, connection_id: str, message: WebSocketMessage):
        """
        Handle pong response from client.
        
        Args:
            connection_id: Connection identifier
            message: Pong message
        """
        if connection_id in self.pending_pings:
            ping_time = self.pending_pings.pop(connection_id)
            response_time = time.time() - ping_time
            
            logger.debug(
                f"Pong received from {connection_id}: {response_time:.3f}s",
                extra={
                    "connection_id": connection_id,
                    "response_time_ms": response_time * 1000
                }
            )
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """
        Get information about a connection.
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            Connection information or None if not found
        """
        return self.connection_info.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """
        Get all connection IDs for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of connection IDs
        """
        return list(self.user_connections.get(user_id, set()))
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connections)
    
    def get_user_count(self) -> int:
        """Get number of unique connected users."""
        return len(self.user_connections)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get connection manager statistics.
        
        Returns:
            Dictionary with various statistics
        """
        total_messages_sent = sum(stats["messages_sent"] for stats in self.connection_stats.values())
        total_messages_received = sum(stats["messages_received"] for stats in self.connection_stats.values())
        
        return {
            "total_connections": len(self.connections),
            "unique_users": len(self.user_connections),
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "pending_pings": len(self.pending_pings),
            "average_messages_per_connection": {
                "sent": total_messages_sent / len(self.connections) if self.connections else 0,
                "received": total_messages_received / len(self.connections) if self.connections else 0
            }
        }
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._periodic_health_check())
    
    async def _stop_background_tasks(self):
        """Stop background maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    async def _cleanup_stale_connections(self):
        """Periodically clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.utcnow()
                stale_connections = []
                
                # Find connections that haven't been active for 30 minutes
                for connection_id, conn_info in self.connection_info.items():
                    if (current_time - conn_info.last_activity).total_seconds() > 1800:  # 30 minutes
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.info(f"Cleaning up stale connection: {connection_id}")
                    await self.disconnect(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _periodic_health_check(self):
        """Periodically ping connections to check health."""
        while True:
            try:
                await asyncio.sleep(120)  # Ping every 2 minutes
                
                # Ping all connections
                for connection_id in list(self.connections.keys()):
                    await self.ping_connection(connection_id)
                
                # Clean up connections that haven't responded to pings
                await asyncio.sleep(30)  # Wait 30 seconds for responses
                
                for connection_id in list(self.pending_pings.keys()):
                    ping_time = self.pending_pings[connection_id]
                    if time.time() - ping_time > 60:  # No response for 1 minute
                        logger.warning(f"Connection {connection_id} not responding to pings, disconnecting")
                        await self.disconnect(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check task: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Global connection manager instance
connection_manager = ConnectionManager()