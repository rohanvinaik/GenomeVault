# GenomeVault Version Tracking

## Current Versions

### PIR Protocol
- **Protocol Version**: PIR-IT-1.0
- **Protocol Revision**: 2025.01.24
- **Compatibility**: Backward compatible with PIR-IT-0.9

### Circuit Versions
- **ZK Circuits**: v2.1.0
- **HDC Encoder**: v1.3.0
- **Verifier Contracts**: v1.2.0

### Encoder Seeds
- **Master Seed**: 0x7a3b9c5d... (truncated for security)
- **HDC Seed**: HDC-2025-01
- **ZK Seed**: ZK-SNARK-2025-01

### Component Versions
- **PIR Client**: 1.0.0
- **PIR Server**: 1.0.0
- **Query Builder**: 1.0.0
- **Shard Manager**: 1.0.0

## Version History

### PIR-IT-1.0 (2025-01-24)
- Initial production release
- 2-server IT-PIR with XOR-based scheme
- Fixed-size padding implementation
- Timing attack mitigations

### PIR-IT-0.9 (2025-01-15)
- Beta release
- Basic query/response functionality
- Performance optimizations

## Migration Notes

### Upgrading from 0.9 to 1.0
1. Update client padding configuration
2. Regenerate shard manifests with new format
3. Update server timing configurations
