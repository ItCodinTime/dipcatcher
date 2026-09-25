"""Research notebooks and docs as fx-1 corpus sources.

The lab's markdown artifacts (research notebooks under
``data/metadata/research/`` and the docs contracts) teach fx-1 how the
harness reasons. Every example carries the SHA-256 of its source file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fx1.data.corpus import SFTExample


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def notebook_examples(
    path: str | Path, system: str, *, chunk_chars: int = 4000
) -> list[SFTExample]:
    """Convert one markdown artifact into teach/explain SFT examples.

    Long documents are chunked on section boundaries so no example exceeds
    *chunk_chars* of source content.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    digest = _sha256_text(text)
    sections = [s for s in text.split("\n#") if s.strip()]
    examples: list[SFTExample] = []
    for i, section in enumerate(sections):
        content = section[:chunk_chars]
        user = (
            f"Explain this dipcatcher harness artifact section and what "
            f"evidence class it belongs to:\n\n{content[:1500]}"
        )
        assistant = (
            f"Source hash {digest[:16]}… (section {i + 1}/{len(sections)}). "
            "This is harness documentation: it defines contracts and evidence "
            "classes, not market results. Claims derived from it must cite this "
            "hash and remain verifiable with `uv run dipcatcher verify-research`."
        )
        examples.append(
            SFTExample(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                receipt_sha256=digest,
                source_path=str(source),
            )
        )
    return examples
