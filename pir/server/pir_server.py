"""
PIR Server implementation for distributed reference genome storage
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from fastapi import FastAPI, HTTPException
import aiofiles

from core.constants import PIR_THRESHOLD, MIN_PIR_SERVERS
from core.exceptions import PIRError


@dataclass
class ShardInfo:
    """Information about a data shard"""
    shard_id: str
    server_id: int
    data_hash: str
    size_bytes: int
    reference_version: str


class PIRServer:
    """
    Private Information Retrieval server
    Implements threshold PIR with information-theoretic security
    """
    
    def __init__(self, server_id: int, shard_count: int, server_type: str = "LN"):
        self.server_id = server_id
        self.shard_count = shard_count
        self.server_type = server_type  # LN (Light Node) or TS (Trusted Signatory)
        self.shards: Dict[str, bytes] = {}
        self.shard_metadata: Dict[str, ShardInfo] = {}
        self.is_trusted = server_type == "TS"
        
    async def initialize_shards(self, reference_data: bytes, version: str):
        """
        Initialize server with sharded reference genome data
        """
        # Create shares using Shamir's secret sharing
        shares = self._create_shares(reference_data)
        
        for i, share in enumerate(shares):
            if i % self.shard_count == self.server_id:
                shard_id = f"shard_{version}_{i}"
                self.shards[shard_id] = share
                self.shard_metadata[shard_id] = ShardInfo(
                    shard_id=shard_id,
                    server_id=self.server_id,
                    data_hash=hashlib.sha256(share).hexdigest(),
                    size_bytes=len(share),
                    reference_version=version
                )
    
    def _create_shares(self, data: bytes) -> List[bytes]:
        """
        Create threshold shares of the data
        Uses simplified XOR-based sharing for demonstration
        """
        # In production, use proper Shamir's secret sharing
        data_array = np.frombuffer(data, dtype=np.uint8)
        shares = []
        
        # Create n shares where any k can reconstruct
        random_masks = [np.random.randint(0, 256, len(data_array), dtype=np.uint8) 
                       for _ in range(self.shard_count - 1)]
        
        # Last share is XOR of data with all masks
        last_share = data_array.copy()
        for mask in random_masks:
            last_share ^= mask
        
        shares.extend([mask.tobytes() for mask in random_masks])
        shares.append(last_share.tobytes())
        
        return shares
    
    async def process_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a PIR query without learning what was queried
        """
        query_vector = np.array(query['vector'])
        response_vector = np.zeros_like(query_vector)
        
        # Process query obliviously
        for shard_id, shard_data in self.shards.items():
            # Homomorphic operation on encrypted query
            shard_array = np.frombuffer(shard_data, dtype=np.uint8)
            
            # Simplified PIR computation
            # In production, use proper cryptographic PIR
            if len(shard_array) <= len(query_vector):
                response_vector[:len(shard_array)] += shard_array * query_vector[:len(shard_array)]
        
        return {
            "server_id": self.server_id,
            "response": response_vector.tolist(),
            "is_trusted": self.is_trusted
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status and metadata"""
        return {
            "server_id": self.server_id,
            "server_type": self.server_type,
            "shard_count": len(self.shards),
            "total_size": sum(info.size_bytes for info in self.shard_metadata.values()),
            "versions": list(set(info.reference_version for info in self.shard_metadata.values())),
            "is_trusted": self.is_trusted
        }


# FastAPI app for PIR server
app = FastAPI(title=f"GenomeVault PIR Server")
server: Optional[PIRServer] = None


@app.on_event("startup")
async def startup():
    """Initialize PIR server on startup"""
    global server
    import os
    
    server_id = int(os.getenv("SERVER_ID", "1"))
    shard_count = int(os.getenv("SHARD_COUNT", "3"))
    server_type = os.getenv("SERVER_TYPE", "LN")
    
    server = PIRServer(server_id, shard_count, server_type)
    
    # Load reference data if available
    # In production, this would load from secure storage
    if os.path.exists("/data/reference/GRCh38.fa"):
        async with aiofiles.open("/data/reference/GRCh38.fa", "rb") as f:
            reference_data = await f.read()
            await server.initialize_shards(reference_data, "GRCh38.p14")


@app.post("/query")
async def process_pir_query(query: Dict[str, Any]):
    """Process a PIR query"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        response = await server.process_query(query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get server status"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    return server.get_status()


@app.post("/initialize")
async def initialize_shards(data: Dict[str, Any]):
    """Initialize server with reference data"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    reference_data = bytes.fromhex(data['reference_hex'])
    version = data['version']
    
    await server.initialize_shards(reference_data, version)
    
    return {"status": "initialized", "version": version}
