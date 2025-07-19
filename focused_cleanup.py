#!/usr/bin/env python3
"""
Focused tail-chasing cleanup
Targets specific known phantom imports
"""

import re
from pathlib import Path

def remove_phantom_imports(filepath: Path, phantom_imports: list):
    """Remove specific phantom imports from a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        
        for phantom in phantom_imports:
            # Remove the import
            patterns = [
                # from .module import phantom
                rf'from [.\w]+ import .*\b{re.escape(phantom)}\b.*\n',
                # phantom,
                rf'\b{re.escape(phantom)}\b,\s*\n',
                # phantom
                rf'\b{re.escape(phantom)}\b,?\s*(?=\))',
            ]
            
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE)
        
        # Clean up empty imports
        content = re.sub(r'from [.\w]+ import \(\s*\)', '', content)
        content = re.sub(r',\s*\)', ')', content)
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    print("🎯 Focused Tail-Chasing Cleanup")
    print("=" * 50)
    
    # Known phantom imports
    phantom_map = {
        'hypervector_transform/__init__.py': [
            'bind_vectors',
            'bundle_vectors', 
            'unbind_vectors',
            'permute_vector',
            'circular_convolution',
            'cross_modal_binding',
            'create_random_projection',
            'encode_features',
            'DomainProjection',
            'EncodingConfig',
            'HolographicEncoder',
            'calculate_capacity',
            'create_holographic_memory',
            'retrieve_pattern',
            'store_pattern',
            'DistanceMetric',
            'SimilarityMapper',
            'create_isometric_mapping',
            'map_to_similarity_space',
            'preserve_distances',
            'CompressionProfile',
            'ResolutionLevel',
            'compress_hypervector',
            'encode_hierarchical',
        ],
        'utils/__init__.py': [
            'GenomeVaultLogger',
            'LogEvent',
            'PrivacyLevel',
            'log_genomic_operation',
            'log_operation',
            'configure_logging',
        ],
        'local_processing/__init__.py': [
            'SequencingEngine',  # Should be SequencingProcessor
            'TranscriptomicsEngine',  # Should be TranscriptomicsProcessor
        ],
    }
    
    # Remove phantom imports
    for filepath, phantoms in phantom_map.items():
        path = Path(filepath)
        if path.exists():
            print(f"\n📄 Cleaning {filepath}...")
            if remove_phantom_imports(path, phantoms):
                print(f"  ✅ Removed {len(phantoms)} phantom imports")
            else:
                print(f"  ℹ️  No changes needed")
    
    # Fix specific __init__.py files with correct imports
    fixes = {
        'hypervector_transform/__init__.py': '''"""
GenomeVault Hypervector Transform Package
"""

from .binding import (
    BindingType,
    HypervectorBinder,
    PositionalBinder,
    CrossModalBinder,
    circular_bind,
    protect_vector,
)

from .encoding import (
    ProjectionType,
    HypervectorConfig,
    HypervectorEncoder,
    create_encoder,
    encode_genomic_data,
)

__all__ = [
    # Binding
    'BindingType',
    'HypervectorBinder',
    'PositionalBinder', 
    'CrossModalBinder',
    'circular_bind',
    'protect_vector',
    # Encoding
    'ProjectionType',
    'HypervectorConfig',
    'HypervectorEncoder',
    'create_encoder',
    'encode_genomic_data',
]
''',
        'local_processing/__init__.py': '''"""
GenomeVault Local Processing Package
"""

from .sequencing import (
    SequencingProcessor,
    DifferentialStorage,
    GenomicProfile,
    Variant,
    QualityMetrics,
)

try:
    from .transcriptomics import (
        TranscriptomicsProcessor,
        ExpressionProfile,
        GeneExpression,
    )
except ImportError:
    TranscriptomicsProcessor = None
    ExpressionProfile = None
    GeneExpression = None

try:
    from .epigenetics import (
        EpigeneticsProcessor,
        MethylationProfile,
        MethylationSite,
    )
except ImportError:
    EpigeneticsProcessor = None
    MethylationProfile = None
    MethylationSite = None

__all__ = [
    'SequencingProcessor',
    'DifferentialStorage',
    'GenomicProfile',
    'Variant',
    'QualityMetrics',
]

if TranscriptomicsProcessor:
    __all__.extend(['TranscriptomicsProcessor', 'ExpressionProfile', 'GeneExpression'])
    
if EpigeneticsProcessor:
    __all__.extend(['EpigeneticsProcessor', 'MethylationProfile', 'MethylationSite'])
''',
    }
    
    # Apply fixes
    print("\n" + "=" * 50)
    print("📝 Applying targeted fixes...")
    
    for filepath, content in fixes.items():
        path = Path(filepath)
        if path.exists():
            print(f"\n✍️  Rewriting {filepath}...")
            with open(path, 'w') as f:
                f.write(content)
            print("  ✅ Done")
    
    # Test imports
    print("\n" + "=" * 50)
    print("🧪 Testing imports...")
    
    test_imports = [
        "from core.config import get_config",
        "from utils import get_logger",
        "from local_processing import SequencingProcessor",
        "from hypervector_transform import HypervectorEncoder",
        "from zk_proofs.prover import Prover",
    ]
    
    for imp in test_imports:
        try:
            exec(imp)
            print(f"✅ {imp}")
        except Exception as e:
            print(f"❌ {imp} - {e}")
    
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    main()
