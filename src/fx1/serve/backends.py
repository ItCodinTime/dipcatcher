"""Inference backends for fx-1.

Two backends, one interface:
- ``HostedK3Backend`` — Moonshot's hosted K3 API. Serves as fx-1's teacher
  (tool-use trace generation) and as the day-one serving path.
- ``LocalFx1Backend`` — a fine-tuned fx-1 checkpoint. Fail-closed until a
  checkpoint with a valid model card exists on disk.

Credentials come from environment variables only — never hardcoded.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Protocol

from fx1.modelcard import ModelCard

MOONSHOT_API_URL = "https://api.moonshot.ai/v1/chat/completions"


class InferenceBackend(Protocol):
    """Chat-completion interface shared by all fx-1 backends."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class HostedK3Backend:
    """Hosted Kimi K3 via the Moonshot API (stdlib HTTP; no new deps)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k3",
        api_url: str = MOONSHOT_API_URL,
    ) -> None:
        self._api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials")
        self._model = model
        self._api_url = api_url

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": self._model, "messages": messages}).encode()
        request = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            payload = json.loads(response.read().decode())
        return str(payload["choices"][0]["message"]["content"])


class LocalFx1Backend:
    """A local fx-1 checkpoint. Fail-closed without a valid model card.

    Attestation enforcement: when ``require_signature=True`` (the default once
    FX1_SIGNING_KEY is set), unsigned or signature-mismatched checkpoints
    refuse to serve — the served model must be the model whose card passed
    the gates, verifiably.
    """

    def __init__(
        self, checkpoint_dir: str | Path, *, require_signature: bool | None = None
    ) -> None:
        root = Path(checkpoint_dir)
        card_path = root / "modelcard.json"
        if not card_path.exists():
            raise FileNotFoundError(
                f"no model card at {card_path}; an fx-1 checkpoint without a card is not servable"
            )
        self.card = ModelCard.load(card_path)
        if not self.card.eval_delta.ship_eligible:
            raise RuntimeError(
                f"{self.card.version} failed the ship gate "
                "(honesty/domain/general deltas); refusing to serve"
            )
        if require_signature is None:
            require_signature = bool(os.environ.get("FX1_SIGNING_KEY"))
        if require_signature:
            from fx1.serve.signing import verify_release

            if not verify_release(root):
                raise RuntimeError(
                    "checkpoint release signature missing or invalid; "
                    "fx-1 serves only signed releases when FX1_SIGNING_KEY "
                    "is configured"
                )
        self._root = root

    def complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError(
            "local inference requires the serving stack (vLLM/SGLang with K3 "
            "support); wire it here when the first distilled student ships"
        )


def get_backend(kind: str, **kwargs: object) -> InferenceBackend:
    """Backend factory: ``hosted_k3`` or ``local_fx1``. Fail-closed."""
    backends = {"hosted_k3": HostedK3Backend, "local_fx1": LocalFx1Backend}
    if kind not in backends:
        raise KeyError(f"unknown backend {kind!r}; choose from {sorted(backends)}")
    return backends[kind](**kwargs)  # type: ignore[arg-type, return-value]
