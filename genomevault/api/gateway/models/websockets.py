"""
WebSocket models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel


class MessageType(str, Enum):
    """WebSocket message types."""

    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"

    # Subscriptions
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    SUBSCRIPTION_ERROR = "subscription_error"

    # Data updates
    DATA_UPDATE = "data_update"
    STATUS_UPDATE = "status_update"
    NOTIFICATION = "notification"
    ALERT = "alert"

    # Errors
    ERROR = "error"
    WARNING = "warning"


class SubscriptionType(str, Enum):
    """Types of subscriptions available."""

    # Pipeline subscriptions
    PIPELINE_STATUS = "pipeline_status"
    PIPELINE_LOGS = "pipeline_logs"
    PIPELINE_METRICS = "pipeline_metrics"

    # Model training subscriptions
    TRAINING_STATUS = "training_status"
    TRAINING_METRICS = "training_metrics"

    # Query subscriptions
    QUERY_STATUS = "query_status"
    QUERY_RESULTS = "query_results"

    # Algorithm execution subscriptions
    EXECUTION_STATUS = "execution_status"
    EXECUTION_LOGS = "execution_logs"

    # Network topology subscriptions
    TOPOLOGY_CHANGES = "topology_changes"
    NODE_STATUS = "node_status"

    # System subscriptions
    SYSTEM_HEALTH = "system_health"
    SYSTEM_ALERTS = "system_alerts"

    # Audit subscriptions
    AUDIT_EVENTS = "audit_events"
    SECURITY_ALERTS = "security_alerts"


class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""

    message_id: str = Field(..., description="Unique message identifier")
    message_type: MessageType = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")

    # Message content
    data: Optional[Dict[str, Any]] = Field(None, description="Message payload")

    # Metadata
    source: Optional[str] = Field(None, description="Message source")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier for request tracking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message_id": "msg_abc123456789",
                "message_type": "data_update",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {
                    "subscription_id": "sub_def456789012",
                    "update_type": "status_change",
                    "content": {
                        "pipeline_id": "pipeline_123",
                        "status": "completed",
                        "progress": 100
                    }
                },
                "source": "pipeline_manager"
            }
        }


class WebSocketResponse(BaseModel):
    """WebSocket response message model."""

    message_id: str = Field(..., description="Response message identifier")
    request_message_id: Optional[str] = Field(None, description="Original request message ID")
    message_type: MessageType = Field(..., description="Response message type")
    timestamp: datetime = Field(..., description="Response timestamp")

    # Response status
    success: bool = Field(..., description="Whether the operation was successful")

    # Response content
    data: Optional[Dict[str, Any]] = Field(None, description="Response payload")
    error: Optional[str] = Field(None, description="Error message if unsuccessful")
    error_details: Optional[Dict[str, str]] = Field(None, description="Detailed error information")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message_id": "resp_abc123456789",
                "request_message_id": "msg_def456789012",
                "message_type": "subscription_confirmed",
                "timestamp": "2024-01-15T10:30:00Z",
                "success": True,
                "data": {
                    "subscription_id": "sub_ghi789012345",
                    "subscription_type": "pipeline_status",
                    "resource_id": "pipeline_123"
                }
            }
        }


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request model."""

    subscription_type: SubscriptionType = Field(..., description="Type of subscription")
    resource_id: str = Field(..., description="Resource identifier to subscribe to")

    # Subscription options
    filters: Optional[Dict[str, Any]] = Field(None, description="Subscription filters")
    update_frequency: Optional[int] = Field(None, description="Update frequency in seconds")
    include_historical: bool = Field(False, description="Include historical data")

    # Authentication context
    auth_context: Optional[Dict[str, str]] = Field(None, description="Authentication context")

    @field_validator("update_frequency")
    def validate_update_frequency(cls, v):
        """Validate update frequency."""
        if v is not None and (v < 1 or v > 3600):
            raise ValueError("Update frequency must be between 1 and 3600 seconds")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": {
                "pipeline_status": {
                    "summary": "Subscribe to pipeline status updates",
                    "value": {
                        "subscription_type": "pipeline_status",
                        "resource_id": "pipeline_abc123456789",
                        "update_frequency": 5,
                        "include_historical": False
                    }
                },
                "system_health": {
                    "summary": "Subscribe to system health updates",
                    "value": {
                        "subscription_type": "system_health",
                        "resource_id": "*",
                        "filters": {
                            "services": ["database", "pir_engine"],
                            "severity": ["warning", "error"]
                        },
                        "update_frequency": 30
                    }
                }
            }
        }


class SubscriptionResponse(BaseModel):
    """WebSocket subscription response model."""

    subscription_id: str = Field(..., description="Unique subscription identifier")
    subscription_type: SubscriptionType = Field(..., description="Type of subscription")
    resource_id: str = Field(..., description="Resource being monitored")

    # Subscription status
    status: str = Field(..., description="Subscription status (active, failed, etc.)")
    created_at: datetime = Field(..., description="Subscription creation time")
    expires_at: Optional[datetime] = Field(None, description="Subscription expiration time")

    # Configuration
    applied_filters: Optional[Dict[str, Any]] = Field(None, description="Applied filters")
    update_frequency: Optional[int] = Field(None, description="Update frequency in seconds")

    # Initial data (if requested)
    initial_data: Optional[Dict[str, Any]] = Field(None, description="Initial data snapshot")

    model_config = {
        "json_schema_extra": {
            "example": {
                "subscription_id": "sub_abc123456789",
                "subscription_type": "pipeline_status",
                "resource_id": "pipeline_abc123456789",
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "update_frequency": 5,
                "initial_data": {
                    "current_status": "running",
                    "progress": 45,
                    "current_step": "hypervector_encoding"
                }
            }
        }


class DataUpdateMessage(BaseModel):
    """Data update message for subscriptions."""

    subscription_id: str = Field(..., description="Subscription identifier")
    update_type: str = Field(..., description="Type of update")
    sequence_number: int = Field(..., description="Sequential update number")

    # Update content
    content: Dict[str, Any] = Field(..., description="Update content")

    # Change information
    changed_fields: Optional[List[str]] = Field(None, description="List of changed fields")
    previous_values: Optional[Dict[str, Any]] = Field(None, description="Previous values for changed fields")

    # Metadata
    source_timestamp: datetime = Field(..., description="Source timestamp of the change")

    model_config = {
        "json_schema_extra": {
            "example": {
                "subscription_id": "sub_abc123456789",
                "update_type": "status_change",
                "sequence_number": 42,
                "content": {
                    "pipeline_id": "pipeline_abc123456789",
                    "status": "completed",
                    "progress": 100,
                    "completed_at": "2024-01-15T10:45:00Z"
                },
                "changed_fields": ["status", "progress", "completed_at"],
                "previous_values": {
                    "status": "running",
                    "progress": 95
                },
                "source_timestamp": "2024-01-15T10:45:00Z"
            }
        }


class NotificationMessage(BaseModel):
    """Notification message model."""

    notification_id: str = Field(..., description="Notification identifier")
    notification_type: str = Field(..., description="Type of notification")
    severity: str = Field(
        ...,
        pattern=r"^(info|warning|error|critical)$",
        description="Notification severity"
    )

    # Notification content
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

    # Targeting
    target_subscriptions: Optional[List[str]] = Field(None, description="Target subscription IDs")
    target_users: Optional[List[str]] = Field(None, description="Target user IDs")

    # Actions
    action_required: bool = Field(False, description="Whether action is required")
    action_url: Optional[str] = Field(None, description="URL for required action")
    expires_at: Optional[datetime] = Field(None, description="Notification expiration")

    model_config = {
        "json_schema_extra": {
            "example": {
                "notification_id": "notif_abc123456789",
                "notification_type": "pipeline_failure",
                "severity": "error",
                "title": "Pipeline Execution Failed",
                "message": "Pipeline 'Genomic Analysis' failed at step 'hypervector_encoding'",
                "details": {
                    "pipeline_id": "pipeline_abc123456789",
                    "execution_id": "exec_def456789012",
                    "failed_step": "hypervector_encoding",
                    "error_code": "INSUFFICIENT_MEMORY"
                },
                "action_required": True,
                "action_url": "/pipelines/pipeline_abc123456789/restart"
            }
        }


class AlertMessage(BaseModel):
    """Alert message model for critical notifications."""

    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(
        ...,
        pattern=r"^(low|medium|high|critical)$",
        description="Alert severity"
    )

    # Alert content
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")

    # Source information
    source_system: str = Field(..., description="System that generated the alert")
    source_component: Optional[str] = Field(None, description="Component that generated the alert")

    # Timing
    triggered_at: datetime = Field(..., description="Alert trigger timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution timestamp")

    # Context
    context: Dict[str, Any] = Field(..., description="Alert context and metadata")

    # Response information
    acknowledged: bool = Field(False, description="Whether alert has been acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged alert")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "alert_id": "alert_abc123456789",
                "alert_type": "security_breach_attempt",
                "severity": "high",
                "title": "Multiple Failed Authentication Attempts",
                "description": "Detected 10 failed authentication attempts from IP 192.168.1.100 within 5 minutes",
                "source_system": "authentication_service",
                "source_component": "oauth2_handler",
                "triggered_at": "2024-01-15T10:30:00Z",
                "context": {
                    "ip_address": "192.168.1.100",
                    "failed_attempts": 10,
                    "time_window_minutes": 5,
                    "attempted_endpoints": ["/auth/token", "/auth/refresh"]
                },
                "acknowledged": False
            }
        }


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""

    connection_id: str = Field(..., description="Connection identifier")
    user_id: Optional[str] = Field(None, description="Authenticated user ID")
    client_info: Optional[Dict[str, str]] = Field(None, description="Client information")

    # Connection details
    connected_at: datetime = Field(..., description="Connection establishment time")
    last_activity: datetime = Field(..., description="Last activity timestamp")

    # Subscriptions
    active_subscriptions: List[str] = Field(..., description="Active subscription IDs")
    subscription_count: int = Field(..., description="Number of active subscriptions")

    # Statistics
    messages_sent: int = Field(..., description="Messages sent to client")
    messages_received: int = Field(..., description="Messages received from client")

    model_config = {
        "json_schema_extra": {
            "example": {
                "connection_id": "conn_abc123456789",
                "user_id": "user_def456789012",
                "client_info": {
                    "user_agent": "GenomeVault-Client/1.0.0",
                    "ip_address": "192.168.1.100"
                },
                "connected_at": "2024-01-15T10:00:00Z",
                "last_activity": "2024-01-15T10:30:00Z",
                "active_subscriptions": ["sub_123", "sub_456"],
                "subscription_count": 2,
                "messages_sent": 47,
                "messages_received": 12
            }
        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
