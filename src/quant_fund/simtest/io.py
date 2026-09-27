"""In-memory disk and HTTP used only by the simulation session.

Neither type performs real network or host-disk IO. Disk-full and corrupt
reads are explicit fault results. The paper ledger still uses the host
filesystem under a temporary directory; these types cover the research cache
and the simulated research API.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from quant_fund.simtest.runtime import DeterministicRuntime


class DiskFull(OSError):
    """Simulated ENOSPC. No bytes were committed."""

    def __init__(self, path: str) -> None:
        super().__init__(28, f"simulated disk full: {path}")
        self.path = path


@dataclass(frozen=True)
class NetResponse:
    status: int
    body: bytes
    latency_us: int

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "body": self.body,
            "latency_us": self.latency_us,
        }


class SimDisk:
    """Byte store with optional forced ENOSPC and single-byte corruption."""

    def __init__(self, runtime: DeterministicRuntime) -> None:
        self._runtime = runtime
        self._files: dict[str, bytes] = {}

    def write(self, path: str, data: bytes, *, force_full: bool = False) -> None:
        digest = hashlib.sha256(data).hexdigest()
        if force_full:
            self._runtime.effect(
                "disk_write",
                {"path": path, "sha256": digest, "full": True},
                {"ok": False, "error": "ENOSPC"},
            )
            raise DiskFull(path)
        self._files[path] = bytes(data)
        self._runtime.effect(
            "disk_write",
            {"path": path, "sha256": digest, "full": False},
            {"ok": True, "error": ""},
        )

    def read(self, path: str, *, corrupt: bool = False) -> bytes:
        data = self._files[path]
        if corrupt and data:
            flipped = bytearray(data)
            flipped[0] ^= 0xFF
            data = bytes(flipped)
        recorded = self._runtime.effect(
            "disk_read",
            {"path": path, "corrupt": corrupt},
            {"sha256": hashlib.sha256(data).hexdigest(), "data": data},
        )
        raw = recorded["data"]
        if isinstance(raw, bytes):
            return raw
        raise RuntimeError("disk read trace did not return bytes")


class SimNetwork:
    """One simulated research endpoint. Slow responses still return a body."""

    def __init__(self, runtime: DeterministicRuntime) -> None:
        self._runtime = runtime

    def request(
        self,
        method: str,
        url: str,
        body: bytes,
        *,
        status: int,
        slow: bool,
    ) -> NetResponse:
        latency_us = 5_000_000 if slow else 100
        payload = b'{"ok":true,"research_only":true}' if status < 500 else b'{"ok":false}'
        recorded = self._runtime.effect(
            "net",
            {
                "method": method,
                "url": url,
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "status": status,
                "slow": slow,
            },
            {"status": status, "latency_us": latency_us, "body": payload},
        )
        raw_body = recorded["body"]
        if not isinstance(raw_body, bytes):
            raise RuntimeError("network trace did not return bytes")
        return NetResponse(
            status=int(recorded["status"]),
            body=raw_body,
            latency_us=int(recorded["latency_us"]),
        )
