import hashlib
import importlib.util
import lzma
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fetch_cnn_zoo_fixture", ROOT / "scripts/fetch_cnn_zoo_fixture.py")
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def test_decode_last_block_preserves_range_in_staging_directory(tmp_path, monkeypatch):
    payload = b"CNN Zoo fixture payload\n"
    stream = lzma.compress(payload, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64)
    header_size = footer_size = 12
    index_size = (int.from_bytes(stream[-8:-4], "little") + 1) * 4
    block_end = len(stream) - footer_size - index_size
    block_bytes = stream[header_size:block_end]
    prefix, block, suffix = (tmp_path / f"{name}.xz" for name in ("prefix", "block", "suffix"))
    prefix.write_bytes(stream[:header_size])
    block.write_bytes(block_bytes)
    suffix.write_bytes(stream[block_end:])
    digest = hashlib.sha256(block_bytes).hexdigest()
    monkeypatch.setitem(FETCH.RANGES, "block", (header_size, block_end - 1, digest))

    decoded, offset = FETCH.decode_last_block(prefix, block, suffix, tmp_path)

    assert block.read_bytes() == block_bytes
    assert decoded.read_bytes() == payload
    assert offset == 0
