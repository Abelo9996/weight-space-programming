"""Fetch two CC BY 4.0 Small CNN Zoo fixtures from the pinned authors' archive."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zlib

import numpy as np


SOURCE = "https://storage.googleapis.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz?generation=1617698236426937"
ARCHIVE_BYTES = 4_197_483_628
ARCHIVE_MD5 = "bfb306baa78639e1f093e0b1007750f5"
RANGES = {
    "prefix": (0, 16_777_215, "157275c2d7487161d7d1bf795a4e3a7ac83dcdc2c9fea54ac9d3aaaeb89ccd07"),
    "block": (4_080_137_740, 4_197_483_391, "100be62f19d72fb432e93977953acd6071abd32d74e3fb59ebe650123b5a02f2"),
    "suffix": (4_197_418_092, 4_197_483_627, "03f1ffece41b3e3df4f64574cd68b4292fc47b6ac60806e000ddc78a922053f7"),
}
MIN_ACCURACY = 0.20
RANGE_QUERIES = {"prefix": "", "block": "&slice=lastblock", "suffix": "&slice=tail"}
CHUNK_SIZE = 1024 * 1024


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_ranges(directory):
    for name, (start, end, _) in RANGES.items():
        print(f"Fetching {name} ({end - start + 1:,} bytes)", flush=True)
        request = Request(SOURCE + RANGE_QUERIES[name], headers={"Range": f"bytes={start}-{end}"})
        expected_range = f"bytes {start}-{end}/{ARCHIVE_BYTES}"
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"unexpected Content-Range for {name}")
            remaining = end - start + 1
            with (directory / f"{name}.xz").open("xb") as target:
                while remaining:
                    chunk = response.read(min(remaining, CHUNK_SIZE))
                    if not chunk:
                        raise ValueError(f"truncated {name} range")
                    target.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError(f"oversized {name} range")


def crc32_bytes(data):
    return struct.pack("<I", zlib.crc32(data))


def decode_vli(data, position):
    value = 0
    for shift in range(0, 63, 7):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, position
    raise ValueError("malformed XZ variable-length integer")


def encode_vli(value):
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    return encoded + bytes([value])


def align(value, boundary):
    return (value + boundary - 1) // boundary * boundary


def decode_last_block(prefix, block, suffix, directory):
    """Wrap the downloaded block as a complete XZ stream and validate its CRC64."""
    ending = suffix.read_bytes()
    footer = ending[-12:]
    if footer[-2:] != b"YZ" or footer[:4] != crc32_bytes(footer[4:10]):
        raise ValueError("invalid XZ footer")
    index_size = (struct.unpack("<I", footer[4:8])[0] + 1) * 4
    index = ending[-12 - index_size:-12]
    if index[0] != 0 or index[-4:] != crc32_bytes(index[:-4]):
        raise ValueError("invalid XZ index")
    count, position = decode_vli(index, 1)
    records = []
    for _ in range(count):
        compressed, position = decode_vli(index, position)
        uncompressed, position = decode_vli(index, position)
        records.append((compressed, uncompressed))

    compressed, uncompressed = records[-1]
    offset = sum(size for _, size in records[:-1])
    block_start = 12 + sum(align(size, 4) for size, _ in records[:-1])
    if (block_start, block_start + align(compressed, 4) - 1) != RANGES["block"][:2]:
        raise ValueError("final-block range does not match original XZ index")

    short_index = bytearray(b"\0\1") + encode_vli(compressed) + encode_vli(uncompressed)
    short_index += bytes(align(len(short_index), 4) - len(short_index))
    short_index += crc32_bytes(short_index)
    footer_body = struct.pack("<I", len(short_index) // 4 - 1) + footer[8:10]

    decoded = directory / "block.tar-slice"
    with tempfile.TemporaryFile(mode="w+b", dir=directory) as reconstructed:
        with prefix.open("rb") as first, block.open("rb") as source:
            reconstructed.write(first.read(12))
            shutil.copyfileobj(source, reconstructed)
        reconstructed.write(short_index + crc32_bytes(footer_body) + footer_body + b"YZ")
        reconstructed.seek(0)
        with lzma.open(reconstructed) as source, decoded.open("wb") as target:
            shutil.copyfileobj(source, target)
    if decoded.stat().st_size != uncompressed:
        raise ValueError("unexpected decoded XZ block length")
    return decoded, offset


def read_prefix(prefix):
    with lzma.open(prefix) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if member.name != "cifar10/weights.npy":
                continue
            numpy_file = archive.extractfile(member)
            if np.lib.format.read_magic(numpy_file) != (1, 0):
                raise ValueError("unexpected NumPy file version")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(numpy_file)
            if shape != (270_000, 4_970) or fortran or dtype.str != "<f4":
                raise ValueError("unexpected archive weights format")
            return member.offset_data + numpy_file.tell(), shape, dtype, member.offset_data + align(member.size, 512)
    raise ValueError("weights.npy absent from archive prefix")


def export_fixtures(decoded, block_offset, weight_info, output):
    weight_start, shape, dtype, annotation_start = weight_info
    row_bytes = shape[1] * dtype.itemsize
    first_complete_row = align(block_offset - weight_start, row_bytes) // row_bytes
    with decoded.open("rb") as stream:
        stream.seek(annotation_start - block_offset)
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            layout_member = archive.next()
            if layout_member.name != "cifar10/layout.csv":
                raise ValueError("unexpected layout member")
            layout = archive.extractfile(layout_member).read()
            metrics_member = archive.next()
            if metrics_member.name != "cifar10/metrics.csv.gz":
                raise ValueError("unexpected metrics member")
            metrics = archive.extractfile(metrics_member).read()

    selected = {}
    row_count = 0
    with gzip.open(io.BytesIO(metrics), "rt", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            row_count += 1
            activation = row["config.activation"]
            if (
                index >= first_complete_row
                and activation in ("relu", "tanh")
                and activation not in selected
                and row["step"] == "86"
                and row["laststep"] == "True"
                and float(row["test_accuracy"]) >= MIN_ACCURACY
            ):
                metrics_row = {key: value for key, value in row.items() if key not in {"", "modeldir"}}
                selected[activation] = (index, metrics_row)
        if row_count != shape[0] or set(selected) != {"relu", "tanh"}:
            raise ValueError("unexpected metrics row count or missing activation fixtures")

    output.mkdir(parents=True, exist_ok=True)
    fixtures = {}
    with decoded.open("rb") as source:
        for activation, (index, metrics_row) in sorted(selected.items()):
            uncompressed_offset = weight_start + index * row_bytes
            source.seek(uncompressed_offset - block_offset)
            raw = source.read(row_bytes)
            weights = np.frombuffer(raw, dtype=dtype)
            if weights.shape != (shape[1],) or not np.isfinite(weights).all():
                raise ValueError("unexpected checkpoint array shape or values")
            name = f"cifar10_{activation}_row{index}.npy"
            with (output / name).open("xb") as target:
                np.save(target, weights, allow_pickle=False)
            fixtures[activation] = {
                "path": name,
                "sha256": sha256_file(output / name),
                "weight_payload_sha256": hashlib.sha256(raw).hexdigest(),
                "zero_based_archive_row": index,
                "uncompressed_weight_row_offset": uncompressed_offset,
                "metrics_row": metrics_row,
            }
    with (output / "layout.csv").open("xb") as target:
        target.write(layout)
    manifest = {
        "dataset": "Small CNN Zoo CIFAR-10-GS",
        "license": "CC-BY-4.0",
        "source": SOURCE,
        "advertised_archive_bytes": ARCHIVE_BYTES,
        "advertised_archive_md5_not_recomputed": ARCHIVE_MD5,
        "range_requests": {
            key: {"start": start, "end_inclusive": end, "sha256": digest}
            for key, (start, end, digest) in RANGES.items()
        },
        "weights_member": "cifar10/weights.npy",
        "weights_shape": list(shape),
        "weights_dtype": dtype.str,
        "uncompressed_weight_data_offset": weight_start,
        "final_block_uncompressed_offset": block_offset,
        "first_complete_row_in_final_block": first_complete_row,
        "layout_member": "cifar10/layout.csv",
        "layout_sha256": hashlib.sha256(layout).hexdigest(),
        "metrics_member": "cifar10/metrics.csv.gz",
        "metrics_gzip_sha256": hashlib.sha256(metrics).hexdigest(),
        "selection": "First fully available final-step (86, laststep=True) row per activation with release test accuracy >= 0.20. Used only as a nondegenerate validation fixture.",
        "fixtures": fixtures,
    }
    with (output / "manifest.json").open("x") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/cnn_zoo_fixtures"))
    parser.add_argument("--ranges-dir", type=Path, help="reuse local prefix.xz, block.xz and suffix.xz files")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cnn-zoo-", dir=args.output.parent) as directory:
        staging = Path(directory)
        ranges = staging if args.ranges_dir is None else args.ranges_dir
        if args.ranges_dir is None:
            download_ranges(ranges)
        paths = {name: ranges / f"{name}.xz" for name in RANGES}
        for name, (start, end, digest) in RANGES.items():
            path = paths[name]
            if path.stat().st_size != end - start + 1 or sha256_file(path) != digest:
                raise ValueError(f"{name} range does not match the pinned archive")
        decoded, offset = decode_last_block(paths["prefix"], paths["block"], paths["suffix"], staging)
        extracted = staging / "fixtures"
        export_fixtures(decoded, offset, read_prefix(paths["prefix"]), extracted)
        extracted.rename(args.output)
    print(f"Saved fixtures to {args.output}")


if __name__ == "__main__":
    main()
