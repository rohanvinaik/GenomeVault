"""Package initialization for crypto."""

from .transcript import Transcript
from .rng import secure_bytes, xof, xof_uint_mod
from .commit import H, hexH, TAGS
from .signatures import (
    SignatureManager,
    generate_keypair,
    sign_data,
    verify_signature,
    export_private_key,
    export_public_key,
    import_private_key,
    import_public_key,
    sign_and_encode,
    verify_encoded_signature,
)
from .serialization import (
    be_int,
    bstr,
    varbytes,
    pack_bytes_seq,
    pack_str_list,
    pack_int_list,
    pack_kv_map,
    pack_proof_components,
)
from .proof_io import compress_proof, decompress_proof, MAGIC, VERSION

__all__ = [
    "H",
    "MAGIC",
    "SignatureManager",
    "TAGS",
    "Transcript",
    "VERSION",
    "be_int",
    "bstr",
    "compress_proof",
    "decompress_proof",
    "export_private_key",
    "export_public_key",
    "generate_keypair",
    "hexH",
    "import_private_key",
    "import_public_key",
    "pack_bytes_seq",
    "pack_int_list",
    "pack_kv_map",
    "pack_proof_components",
    "pack_str_list",
    "secure_bytes",
    "sign_and_encode",
    "sign_data",
    "varbytes",
    "verify_encoded_signature",
    "verify_signature",
    "xof",
    "xof_uint_mod",
]
