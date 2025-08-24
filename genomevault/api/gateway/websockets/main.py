"""
Main WebSocket router for GenomeVault API Gateway.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPBearer

from genomevault.api.gateway.models.websockets import (
    MessageType,
    WebSocketMessage,
    WebSocketResponse,
    SubscriptionRequest,
)
from genomevault.api.gateway.websockets.connection_manager import connection_manager
from genomevault.observability.logging import get_logger

logger = get_logger(__name__)

websocket_router = APIRouter()

security = HTTPBearer(auto_error=False)


@websocket_router.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = None,
    user_id: str = None
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Args:
        websocket: WebSocket connection
        token: Optional authentication token
        user_id: Optional user identifier
    """
    connection_id = None
    
    try:
        # Extract client information
        client_info = {
            "host": websocket.client.host if websocket.client else "unknown",
            "user_agent": websocket.headers.get("user-agent", "unknown")
        }
        
        # Authenticate user if token provided
        authenticated_user_id = None
        if token:
            authenticated_user_id = await _authenticate_websocket_token(token)
        elif user_id:
            authenticated_user_id = user_id
        
        # Establish connection
        connection_id = await connection_manager.connect(
            websocket=websocket,
            user_id=authenticated_user_id,
            client_info=client_info
        )
        
        logger.info(f"WebSocket client connected: {connection_id}")
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                message = await connection_manager.receive_message(connection_id, websocket)
                
                if message is None:
                    # Connection closed or error
                    break
                
                # Handle the message
                await _handle_websocket_message(connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {connection_id}")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                
                # Send error response to client
                error_response = WebSocketResponse(
                    message_id=str(uuid.uuid4()),
                    request_message_id=getattr(message, 'message_id', None),
                    message_type=MessageType.ERROR,
                    timestamp=datetime.utcnow(),
                    success=False,
                    error=f"Message processing error: {str(e)}"
                )
                await connection_manager.send_response(connection_id, error_response)
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        if connection_id:
            await connection_manager.disconnect(connection_id)


async def _authenticate_websocket_token(token: str) -> str:
    """
    Authenticate WebSocket connection using token.
    
    Args:
        token: Authentication token
        
    Returns:
        User ID if authentication successful, None otherwise
    """
    # TODO: Implement actual token validation
    # This would validate JWT tokens or API keys
    
    if token.startswith("demo_"):
        return f"user_{token[5:13]}"
    
    return None


async def _handle_websocket_message(connection_id: str, message: WebSocketMessage):
    """
    Handle incoming WebSocket message.
    
    Args:
        connection_id: Connection identifier
        message: Incoming message
    """
    try:
        if message.message_type == MessageType.PING:
            # Respond to ping with pong
            pong_response = WebSocketResponse(
                message_id=str(uuid.uuid4()),
                request_message_id=message.message_id,
                message_type=MessageType.PONG,
                timestamp=datetime.utcnow(),
                success=True,
                data=message.data
            )
            await connection_manager.send_response(connection_id, pong_response)
        
        elif message.message_type == MessageType.PONG:
            # Handle pong response
            await connection_manager.handle_pong(connection_id, message)
        
        elif message.message_type == MessageType.SUBSCRIBE:
            # Handle subscription request
            await _handle_subscription_request(connection_id, message)
        
        elif message.message_type == MessageType.UNSUBSCRIBE:
            # Handle unsubscription request
            await _handle_unsubscription_request(connection_id, message)
        
        else:
            logger.warning(f"Unhandled message type: {message.message_type}")
            
            # Send error response
            error_response = WebSocketResponse(
                message_id=str(uuid.uuid4()),
                request_message_id=message.message_id,
                message_type=MessageType.ERROR,
                timestamp=datetime.utcnow(),
                success=False,
                error=f"Unhandled message type: {message.message_type}"
            )
            await connection_manager.send_response(connection_id, error_response)
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        raise


async def _handle_subscription_request(connection_id: str, message: WebSocketMessage):
    """
    Handle subscription request.
    
    Args:
        connection_id: Connection identifier
        message: Subscription message
    """
    try:
        # Parse subscription request from message data
        if not message.data:
            raise ValueError("Subscription request data is required")
        
        subscription_request = SubscriptionRequest(**message.data)
        
        # TODO: Implement subscription logic
        # This would integrate with the subscription manager
        
        # Mock successful subscription response
        success_response = WebSocketResponse(
            message_id=str(uuid.uuid4()),
            request_message_id=message.message_id,
            message_type=MessageType.SUBSCRIPTION_CONFIRMED,
            timestamp=datetime.utcnow(),
            success=True,
            data={
                "subscription_id": f"sub_{int(datetime.utcnow().timestamp() * 1000000)}",
                "subscription_type": subscription_request.subscription_type,
                "resource_id": subscription_request.resource_id
            }
        )
        
        await connection_manager.send_response(connection_id, success_response)
        
        logger.info(
            f"Subscription created for {connection_id}",
            extra={
                "connection_id": connection_id,
                "subscription_type": subscription_request.subscription_type,
                "resource_id": subscription_request.resource_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error handling subscription request: {e}")
        
        error_response = WebSocketResponse(
            message_id=str(uuid.uuid4()),
            request_message_id=message.message_id,
            message_type=MessageType.SUBSCRIPTION_ERROR,
            timestamp=datetime.utcnow(),
            success=False,
            error=f"Subscription failed: {str(e)}"
        )
        await connection_manager.send_response(connection_id, error_response)


async def _handle_unsubscription_request(connection_id: str, message: WebSocketMessage):
    """
    Handle unsubscription request.
    
    Args:
        connection_id: Connection identifier
        message: Unsubscription message
    """
    try:
        # TODO: Implement unsubscription logic
        
        success_response = WebSocketResponse(
            message_id=str(uuid.uuid4()),
            request_message_id=message.message_id,
            message_type=MessageType.SUBSCRIPTION_CONFIRMED,
            timestamp=datetime.utcnow(),
            success=True,
            data={"unsubscribed": True}
        )
        
        await connection_manager.send_response(connection_id, success_response)
        
    except Exception as e:
        logger.error(f"Error handling unsubscription request: {e}")
        
        error_response = WebSocketResponse(
            message_id=str(uuid.uuid4()),
            request_message_id=message.message_id,
            message_type=MessageType.ERROR,
            timestamp=datetime.utcnow(),
            success=False,
            error=f"Unsubscription failed: {str(e)}"
        )
        await connection_manager.send_response(connection_id, error_response)