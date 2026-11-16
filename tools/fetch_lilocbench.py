#!/usr/bin/env python3
"""Fetch a thinned subset of the LILocBench indoor localization dataset.

LILocBench (Uni Bonn, IROS 2025): an office robot with 2D lidars, RealSense
D455 RGB-D cameras, wheel odometry and ceiling-tag ground truth. One mapping
run plus later runs with people and rearranged furniture.

    https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/

License: CC BY-NC-SA 4.0 (non-commercial). The data is not redistributed with
this repo; this tool only downloads it for local evaluation.

A full sequence is 4-18 GB because it holds three cameras at 15 Hz. Vision
here runs at about 2 Hz from one camera, so this tool reads the zip's index
over HTTP range requests and pulls only the frames it needs, plus every small
file (lidar, odometry, calibration) and the ground-truth poses.

    python tools/fetch_lilocbench.py                       # default subset
    python tools/fetch_lilocbench.py --sequences dynamics_0 --rate-hz 1
    python tools/fetch_lilocbench.py --list                # show zip contents

Re-running skips files that are already on disk.
"""

from __future__ import annotations

import argparse
import bisect
import http.client
import io
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_URL = "https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data"
# Sequences whose ground truth is public.
PUBLIC_SEQUENCES = (
    "mapping",
    "static_0",
    "dynamics_0",
    "lt_changes_0",
    "lt_changes_dynamics_0",
)
DEFAULT_SEQUENCES = ("mapping", "dynamics_0", "lt_changes_0")
DEFAULT_OUT = Path.home() / "datasets" / "lilocbench"
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
# Room for the local header's extra field, which can differ from the index's.
_HEADER_SLACK = 512


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file, fetched in cached blocks."""

    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.url = url
        self._block = int(block)
        self._pos = 0
        self._cache: Dict[int, bytes] = {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])
            if "bytes" not in str(resp.headers.get("Accept-Ranges", "")).lower():
                raise RuntimeError(f"{url} does not accept range requests")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def _get_block(self, index: int) -> bytes:
        cached = self._cache.get(index)
        if cached is None:
            start = index * self._block
            end = min(self.size, start + self._block) - 1
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                cached = resp.read()
            self._cache[index] = cached
        return cached

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        out = bytearray()
        while n > 0:
            index, offset = divmod(self._pos, self._block)
            piece = self._get_block(index)[offset : offset + n]
            if not piece:
                break
            out += piece
            self._pos += len(piece)
            n -= len(piece)
        return bytes(out)


class _RangeClient:
    """One keep-alive HTTPS connection per worker thread."""

    def __init__(self, url: str) -> None:
        parts = urllib.parse.urlsplit(url)
        self._host = parts.netloc
        self._path = parts.path
        self._https = parts.scheme == "https"
        self._local = threading.local()

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            conn = cls(self._host, timeout=120)
            self._local.conn = conn
        return conn

    def fetch(self, start: int, length: int, retries: int = 4) -> bytes:
        last: Optional[Exception] = None
        for attempt in range(retries):
            try:
                conn = self._conn()
                conn.request(
                    "GET",
                    self._path,
                    headers={"Range": f"bytes={start}-{start + length - 1}"},
                )
                resp = conn.getresponse()
                body = resp.read()
                if resp.status != 206:
                    raise RuntimeError(f"HTTP {resp.status} for range request")
                return body
            except Exception as exc:  # noqa: BLE001
                last = exc
                try:
                    self._local.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                self._local.conn = None
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"range fetch failed after {retries} tries: {last}")


def extract_member(client: _RangeClient, info: zipfile.ZipInfo, total_size: int) -> bytes:
    """Download and decompress one zip member with a single range request."""
    name_len = len(info.filename.encode("utf-8"))
    want = _LOCAL_HEADER.size + name_len + _HEADER_SLACK + int(info.compress_size)
    want = min(want, total_size - int(info.header_offset))
    raw = client.fetch(int(info.header_offset), want)
    sig, _ver, _flag, method, _t, _d, _crc, _cs, _us, nlen, elen = _LOCAL_HEADER.unpack(
        raw[: _LOCAL_HEADER.size]
    )
    if sig != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {info.filename}")
    data_start = _LOCAL_HEADER.size + nlen + elen
    data_end = data_start + int(info.compress_size)
    if data_end > len(raw):
        raw += client.fetch(int(info.header_offset) + len(raw), data_end - len(raw))
    payload = raw[data_start:data_end]
    if method == zipfile.ZIP_STORED:
        data = payload
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise RuntimeError(f"unsupported zip method {method} for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != int(info.CRC):
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    return data


def _stamp(name: str) -> Optional[float]:
    stem = name.rsplit("/", 1)[-1]
    if not stem.endswith(".png"):
        return None
    try:
        return float(stem[: -len(".png")])
    except ValueError:
        return None


def choose_members(
    infos: Sequence[zipfile.ZipInfo],
    *,
    cameras: Sequence[str],
    rate_hz: float,
    max_pair_gap_s: float = 0.05,
) -> Tuple[List[zipfile.ZipInfo], Dict[str, int]]:
    """Every small file, plus colour/depth pairs thinned to ``rate_hz``.

    A colour frame is kept only when a depth frame from the same camera sits
    within ``max_pair_gap_s`` of it.
    """
    keep: List[zipfile.ZipInfo] = []
    images: Dict[Tuple[str, str], List[Tuple[float, zipfile.ZipInfo]]] = {}
    for info in infos:
        if info.is_dir():
            continue
        parts = info.filename.split("/")
        is_image = (
            len(parts) == 4
            and parts[0].startswith("camera_")
            and parts[1] in ("color", "depth")
            and parts[2] == "images"
        )
        if not is_image:
            keep.append(info)
            continue
        cam = parts[0][len("camera_") :]
        if cam not in cameras:
            continue
        stamp = _stamp(info.filename)
        if stamp is not None:
            images.setdefault((cam, parts[1]), []).append((stamp, info))

    counts: Dict[str, int] = {}
    period = 1.0 / max(1e-6, float(rate_hz))
    for cam in cameras:
        color = sorted(images.get((cam, "color"), []), key=lambda row: row[0])
        depth = sorted(images.get((cam, "depth"), []), key=lambda row: row[0])
        depth_stamps = [row[0] for row in depth]
        last = -float("inf")
        pairs = 0
        for stamp, info in color:
            if stamp - last < period - 1e-3 or not depth:
                continue
            i = bisect.bisect_left(depth_stamps, stamp)
            best = min(
                (j for j in (i - 1, i) if 0 <= j < len(depth)),
                key=lambda j: abs(depth_stamps[j] - stamp),
            )
            if abs(depth_stamps[best] - stamp) > max_pair_gap_s:
                continue
            keep.append(info)
            keep.append(depth[best][1])
            last = stamp
            pairs += 1
        counts[cam] = pairs
    return keep, counts


def _download_plain(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def fetch_sequence(
    name: str,
    out_root: Path,
    *,
    cameras: Sequence[str],
    rate_hz: float,
    workers: int,
    list_only: bool = False,
) -> None:
    url = f"{BASE_URL}/data_files_zipped/{name}.zip"
    remote = HttpRangeFile(url)
    infos = zipfile.ZipFile(remote).infolist()
    if list_only:
        dirs: Dict[str, List[int]] = {}
        for info in infos:
            folder = info.filename.rsplit("/", 1)[0] if "/" in info.filename else "."
            row = dirs.setdefault(folder, [0, 0])
            row[0] += 1
            row[1] += int(info.compress_size)
        print(f"{name}.zip  {remote.size / 1e9:.2f} GB  {len(infos)} members")
        for folder, (n, size) in dirs.items():
            print(f"  {n:6d} files {size / 1e6:9.1f} MB  {folder}")
        return

    members, counts = choose_members(infos, cameras=cameras, rate_hz=rate_hz)
    seq_dir = out_root / name
    todo = [m for m in members if not (seq_dir / m.filename).is_file()]
    size_mb = sum(int(m.compress_size) for m in todo) / 1e6
    print(
        f"{name}: {len(members)} files selected "
        f"({', '.join(f'{c}={n} pairs' for c, n in counts.items())}); "
        f"{len(todo)} to download, {size_mb:.0f} MB"
    )
    client = _RangeClient(url)
    done = 0
    lock = threading.Lock()
    t0 = time.monotonic()

    def _one(info: zipfile.ZipInfo) -> None:
        nonlocal done
        data = extract_member(client, info, remote.size)
        dest = seq_dir / info.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        with lock:
            done += 1
            if done % 200 == 0 or done == len(todo):
                rate = done / max(1e-6, time.monotonic() - t0)
                print(f"  {name}: {done}/{len(todo)}  ({rate:.1f} files/s)", flush=True)

    # Small files first so calibration is on disk even if images are cut short.
    todo.sort(key=lambda m: (m.filename.endswith(".png"), m.filename))
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(_one, todo))

    gt = seq_dir / "gt_poses.txt"
    if not gt.is_file():
        _download_plain(f"{BASE_URL}/gt_poses_public/{name}/gt_poses.txt", gt)
        print(f"  {name}: ground truth -> {gt}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help=f"public-ground-truth sequences: {', '.join(PUBLIC_SEQUENCES)}",
    )
    ap.add_argument("--cameras", nargs="+", default=["front"], choices=["front", "left", "right"])
    ap.add_argument("--rate-hz", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list", action="store_true", help="print zip contents and exit")
    args = ap.parse_args(argv)

    out_root = Path(args.out).expanduser()
    for name in args.sequences:
        if name not in PUBLIC_SEQUENCES:
            print(f"warning: {name} has no public ground truth", file=sys.stderr)
        fetch_sequence(
            name,
            out_root,
            cameras=args.cameras,
            rate_hz=float(args.rate_hz),
            workers=int(args.workers),
            list_only=bool(args.list),
        )
    if not args.list:
        map_zip = out_root / "map_office.zip"
        if not map_zip.is_file():
            _download_plain(f"{BASE_URL}/map_office.zip", map_zip)
            with zipfile.ZipFile(map_zip) as zf:
                zf.extractall(out_root / "map_office")
            print(f"reference occupancy map -> {out_root / 'map_office'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
