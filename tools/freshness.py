#!/usr/bin/env python3
# Adapted from raj200501/raise-v1-public tools/freshness.py (Apache-2.0); changes: claims come
# from the claim registry (artifacts/claims.json) and position markers shared with
# tools/claimcheck.py instead of a regex-per-claim registry; paths are RFC 6901 JSON pointers plus
# named derivations (len, count, sum, mean, min, max) instead of the dotted path grammar; the
# artifact hash recorded in the registry is recomputed; generated code blocks (from=) are
# re-compared; a --regenerate mode rewrites values and hashes; stdlib only.
"""freshness: every live number in a document equals what its artifact says now.

Usage (what `make freshness` runs)::

    python tools/freshness.py README.md docs/REPORT.md

claimcheck asks whether each number is a registered claim. freshness asks whether the claim is
still current. For every claim in the registry (artifacts/claims.json) it recomputes the
artifact's SHA-256, re-reads the JSON pointer, reruns the named derivation, and formats the
result. It fails when:

- an artifact is missing, or its hash differs from the registry's artifact_sha256;
- a pointer or derivation no longer resolves (a failure, never a skip);
- the registry's value differs from the fresh value (the registry is stale);
- a document's text after `<!-- claim:<id> -->` differs from the fresh value;
- a marker names an unknown claim, or a code block with from=artifacts/<file> differs from it.

Last line: `FRESHNESS: PASS <n> live numbers` or `FRESHNESS: FAIL <n> problems`, where n counts
claim markers plus number tokens in generated code blocks. Exit 0 only on PASS.

`--regenerate` rewrites each claim's value and artifact_sha256 from its artifact (keeping every
other field) and sets the registry's generator to tools/freshness.py. It never edits documents.
Standard library only (Python 3.11+).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load_claimcheck() -> ModuleType:
    """Load tools/claimcheck.py by path, so this works without tools/ on sys.path."""
    name = "noisefloor_tools_claimcheck"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "claimcheck.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load tools/claimcheck.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load_claimcheck()


def fresh_values(
    root: Path, claims: dict[str, dict[str, Any]], problems: list[str], registry: str
) -> dict[str, str]:
    """Re-derive every registered claim. Returns {id: fresh value} for those that resolve."""
    fresh: dict[str, str] = {}
    for cid, claim in claims.items():
        try:
            value, digest = cc.derive_claim(root, claim)
        except cc.ClaimError as exc:
            problems.append(f"{registry}: claim {cid}: cannot re-derive: {exc}")
            continue
        if digest != claim.get("artifact_sha256"):
            problems.append(
                f"{registry}: claim {cid}: {claim.get('artifact')} changed since the registry "
                f"was written (sha256 {str(claim.get('artifact_sha256'))[:12]} recorded, "
                f"{digest[:12]} now)"
            )
        if value != claim.get("value"):
            problems.append(
                f"{registry}: claim {cid} is stale: registry says {claim.get('value')!r}, "
                f"{claim.get('artifact')} gives {value!r}"
            )
        fresh[cid] = value
    return fresh


def run(root: Path, docs: Sequence[str], registry: str = cc.REGISTRY) -> tuple[list[str], int, int]:
    """Check documents against fresh values. Returns (output lines, problems, live numbers)."""
    problems: list[str] = []
    claims, reg_problems = cc.load_registry(root, registry)
    problems.extend(reg_problems)
    fresh = fresh_values(root, claims, problems, registry)
    lines = [f"freshness: re-derived {len(fresh)} of {len(claims)} claims in {registry}"]
    live = 0
    for arg in dict.fromkeys(docs):
        path = cc.resolve(root, arg)
        shown = cc.display(root, path)
        if not path.is_file():
            problems.append(f"{shown}: document missing")
            lines.append(f"{shown}: MISSING")
            continue
        doc = cc.scan_document(path.read_text(encoding="utf-8"), shown)
        doc_problems: list[str] = []
        current = 0
        for mk in doc.markers:
            loc = doc.where(mk.offset)
            if mk.pos < 0:
                doc_problems.append(
                    f"{loc}: claim marker {mk.claim_id} does not mark rendered text"
                )
            elif mk.claim_id not in claims:
                doc_problems.append(f"{loc}: claim marker {mk.claim_id} is not in the registry")
            elif mk.claim_id not in fresh:
                doc_problems.append(f"{loc}: claim {mk.claim_id} cannot be re-derived (see above)")
            else:
                wrong = cc.marker_value(doc, mk, fresh[mk.claim_id])
                if wrong is None:
                    current += 1
                else:
                    doc_problems.append(
                        f"{loc}: claim {mk.claim_id}: document shows {wrong!r}, artifact gives "
                        f"{fresh[mk.claim_id]!r} (re-render the document)"
                    )
        block_problems, generated = cc.fence_sources(root, doc)
        doc_problems.extend(block_problems)
        live += current + generated
        problems.extend(doc_problems)
        lines.append(
            f"{shown}: {current} of {len(doc.markers)} claim markers current, {generated} numbers "
            f"in generated blocks, {len(doc_problems)} problems"
        )
    lines.extend(f"  - {p}" for p in problems)
    if problems:
        lines.append(f"FRESHNESS: FAIL {len(problems)} problems")
    else:
        lines.append(f"FRESHNESS: PASS {live} live numbers")
    return lines, len(problems), live


def regenerate(root: Path, registry: str = cc.REGISTRY) -> tuple[list[str], int]:
    """Rewrite value and artifact_sha256 of every claim from its artifact."""
    path = cc.resolve(root, registry)
    if not path.is_file():
        return [f"freshness: {registry} does not exist; write the claim definitions first"], 1
    claims, problems = cc.load_registry(root, registry, definitions_only=True)
    lines = [f"  - {p}" for p in problems]
    if problems:
        lines.append(f"freshness: not rewritten ({len(problems)} problems)")
        return lines, len(problems)
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = 0
    for claim in data["claims"]:
        cid = claim["id"]
        try:
            value, digest = cc.derive_claim(root, claim)
        except cc.ClaimError as exc:
            lines.append(f"  - claim {cid}: cannot derive: {exc}")
            errors += 1
            continue
        if claim.get("value") != value or claim.get("artifact_sha256") != digest:
            lines.append(f"claim {cid}: {claim.get('value')!r} -> {value!r}")
        claim["value"] = value
        claim["artifact_sha256"] = digest
    if errors:
        lines.append(f"freshness: not rewritten ({errors} problems)")
        return lines, errors
    data["generator"] = "tools/freshness.py"
    data.setdefault("schema", cc.SCHEMA)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines.append(f"freshness: rewrote {registry} ({len(claims)} claims)")
    return lines, 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("docs", nargs="*", help="documents to check, relative to the root")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root (default: here)")
    parser.add_argument("--claims", default=cc.REGISTRY, help="claim registry")
    parser.add_argument(
        "--regenerate", action="store_true", help="rewrite registry values and hashes"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.regenerate:
        lines, problems = regenerate(args.root, args.claims)
        print("\n".join(lines))
        return 0 if problems == 0 else 1
    if not args.docs:
        parser.error("name at least one document")
    lines, problems, _ = run(args.root, args.docs, args.claims)
    print("\n".join(lines))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
