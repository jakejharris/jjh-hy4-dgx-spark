#!/usr/bin/env python3
"""Kit integrity gate for the Hy4 four-node release repository."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SLOT_RE = re.compile(r"\{\{(?:claim|value|bare|spread|rows):([a-z0-9_.]+)\}\}")  # the kinds tools/render.py fills

# Build forbidden patterns without embedding literal forbidden strings in this file.
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("home-path", re.compile("/ho" + "me/")),
    ("username", re.compile("jake" + "jh", re.IGNORECASE)),
    ("host-alias", re.compile(r"(?<![A-Za-z0-9])spark" + r"[1-4](?![0-9])", re.IGNORECASE)),
    ("private-tool", re.compile("exlr8" + "enc", re.IGNORECASE)),
    ("git-trailer", re.compile("Co-" + "Authored-" + "By", re.IGNORECASE)),
    ("em-dash", re.compile("\u2014")),
    ("head-placeholder", re.compile(r"<[A-Za-z0-9_-]+-head>")),
    # a home-network address (the first /24 of 192.168); the ring's cable subnets are not matched
    ("home-lan-address", re.compile(r"(?<![0-9])192\.168\." + r"1\.[0-9]")),
]

# The release's own public addresses (kit repository, weights repository), exact prefixes. Text inside them,
# and their exact owner/name form (link text), is not matched against the forbidden patterns, so an owner's
# handle may appear there and nowhere else.
# Empty until the owner names the repositories. The third is the kit repository's raw file address, which the
# model card uses for the README's picture (assets/hero.jpg). Keep the tuple on one line: the publish link test
# reads the line that starts with OWN_URLS.
OWN_URLS: tuple[str, ...] = ("https://github.com/jakejharris/jjh-hy4-dgx-spark", "https://huggingface.co/jakejharris/jjh-hy4-exl3", "https://raw.githubusercontent.com/jakejharris/jjh-hy4-dgx-spark")

# Each shipped patch, and the document that states its sha256.
PATCHES = {
    "serve/patches/0001-hy4-port.patch": "serve/BUILD.md",
    "quant/patches/0001-hy4-quant.patch": "quant/README.md",
}
# Patches get every forbidden rule, plus no mail header.
PATCH_PATTERNS = FORBIDDEN_PATTERNS + [("mail-header", re.compile(r"^From: "))]
# The serve patch skips the host-alias rule for two upstream TensorFold names it matches: the model
# family "spark" + "2_5" (a context line) and the test folder "spark" + "4b-{d}".
HOST_ALIAS_SKIP = "serve/patches/0001-hy4-port.patch"


def iter_kit_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in {".git", "__pycache__"}]
        for name in sorted(filenames):
            if name == ".git":  # a git worktree keeps a .git file pointing at the main repository
                continue
            files.append(Path(dirpath) / name)
    return files


def check_required_files(root: Path, errors: list[str]) -> None:
    for rel in ("NOTICE", "THIRD_PARTY.md"):
        if not (root / rel).is_file():
            errors.append(f"missing required file: {rel}")


def check_forbidden_strings(root: Path, files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in PATCHES:
            continue  # check_patches
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for url in OWN_URLS:  # the release's own addresses, and their owner/name form, may hold the owner's handle
                line = line.replace(url, "").replace(url.split("://", 1)[-1].split("/", 1)[-1], "")
            for label, pat in FORBIDDEN_PATTERNS:
                if pat.search(line):
                    errors.append(f"forbidden {label} in {rel}:{lineno}")


def check_patches(root: Path, errors: list[str]) -> None:
    """Both patches ship, each matches the sha256 its document states, and neither holds a forbidden line."""
    for rel, doc in PATCHES.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"missing patch: {rel}")
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        doc_path = root / doc
        if not doc_path.is_file() or digest not in doc_path.read_text(encoding="utf-8"):
            errors.append(f"{rel}: sha256 {digest} is not the value {doc} states")
        for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            for label, pat in PATCH_PATTERNS:
                if label == "host-alias" and rel == HOST_ALIAS_SKIP:
                    continue
                if pat.search(line):
                    errors.append(f"forbidden {label} in {rel}:{lineno}")


def check_scripts_and_python(root: Path, files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".sh":
            proc = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                msg = proc.stderr.strip() or proc.stdout.strip() or "bash -n failed"
                errors.append(f"shell syntax error in {rel}: {msg}")
        elif path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"python compile error in {rel}: {exc.msg}")


# "interrupted": the run started and did not finish, so the claim has no value; its status_note says why
# and renders where the value would be. A rerun fills it like a pending claim.
# "not_run": the run was planned and dropped before it started (for time, say); same rules as "interrupted".
STATUSES = {"measured", "provisional", "component", "pending", "interrupted", "not_run"}


def check_ledger_and_slots(root: Path, files: list[Path], errors: list[str]) -> str:
    claims_path = root / "card" / "claims.json"
    if not claims_path.is_file():
        return "no ledger yet"

    try:
        data = json.loads(claims_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid card/claims.json: {exc}")
        return "ledger invalid"

    claims = data.get("claims")
    if not isinstance(claims, list):
        errors.append("card/claims.json: 'claims' must be a list")
        return "ledger invalid"

    by_id: dict[str, dict] = {}
    status_counts: Counter[str] = Counter()
    for idx, entry in enumerate(claims):
        if not isinstance(entry, dict):
            errors.append(f"card/claims.json: entry {idx} is not an object")
            continue
        cid = entry.get("id")
        status = entry.get("status", "unknown")
        status_counts[str(status)] += 1
        if not isinstance(cid, str) or not cid:
            errors.append(f"card/claims.json: entry {idx} missing valid 'id'")
            continue
        if cid in by_id:
            errors.append(f"card/claims.json: duplicate claim id '{cid}'")
        by_id[cid] = entry
        if status not in STATUSES:
            errors.append(f"card/claims.json: claim '{cid}' has status {status!r}, not one of {sorted(STATUSES)}")
        if status in ("interrupted", "not_run") and (entry.get("value") is not None or not str(entry.get("status_note") or "").strip()):
            errors.append(f"card/claims.json: {status} claim '{cid}' needs value null and a status_note saying why")
        if status == "measured":
            receipt = entry.get("receipt")
            if not isinstance(receipt, str) or not receipt.strip():
                errors.append(f"card/claims.json: measured claim '{cid}' has no receipt")
            elif not receipt.startswith("S:"):
                rpath = root / receipt
                if not rpath.exists():
                    errors.append(
                        f"card/claims.json: measured claim '{cid}' receipt not found: {receipt}"
                    )
                elif rpath.suffix == ".json":
                    # A receipt that names its claim must name this one (not the other build's).
                    try:
                        named = json.loads(rpath.read_text(encoding="utf-8")).get("claim_id", cid)
                    except (OSError, json.JSONDecodeError, AttributeError):
                        named = cid
                    if named != cid:
                        errors.append(f"card/claims.json: measured claim '{cid}' receipt {receipt} has claim_id '{named}'")

    for path in files:
        if path.suffix != ".md":
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in SLOT_RE.finditer(line):
                cid = match.group(1)
                if cid not in by_id:
                    errors.append(f"unknown claim slot '{cid}' in {rel}:{lineno}")

    parts = [f"{k}={status_counts[k]}" for k in sorted(status_counts)]
    return "claims by status: " + (", ".join(parts) if parts else "none")


TRUNK_WIDTHS = {"4b", "4b6b", "6b"}
TRUNK_PLACEHOLDER = "PICK"
TRUNK_LINE_RE = re.compile(r'^TF_HY4_TRUNK="([^"]*)"\s*$')


def check_trunk_pick(root: Path, errors: list[str]) -> None:
    """The default build's trunk width lives in two places; they must agree."""
    claims_path = root / "card" / "claims.json"
    env_path = root / "launch" / "ring.env.example"
    if not claims_path.is_file() or not env_path.is_file():
        return
    try:
        claims = json.loads(claims_path.read_text(encoding="utf-8")).get("claims", [])
    except (OSError, json.JSONDecodeError):
        return  # reported by check_ledger_and_slots
    entry = next((c for c in claims if isinstance(c, dict) and c.get("id") == "build.full.trunk"), None)
    if entry is None:
        return
    values = [m.group(1) for m in map(TRUNK_LINE_RE.match, env_path.read_text(encoding="utf-8").splitlines()) if m]
    if len(values) != 1:
        errors.append(f"launch/ring.env.example: expected one TF_HY4_TRUNK line, found {len(values)}")
        return
    env_value = values[0]
    pending = entry.get("status", "pending") == "pending" or entry.get("value") is None
    if pending:
        if env_value != TRUNK_PLACEHOLDER:
            errors.append(
                f"launch/ring.env.example: TF_HY4_TRUNK=\"{env_value}\" but claim build.full.trunk is pending "
                f"(set the claim, or keep \"{TRUNK_PLACEHOLDER}\")"
            )
        return
    value = str(entry.get("value"))
    if value not in TRUNK_WIDTHS:
        errors.append(f"card/claims.json: build.full.trunk value '{value}' is not one of {sorted(TRUNK_WIDTHS)}")
    if env_value != value:
        errors.append(
            f"launch/ring.env.example: TF_HY4_TRUNK=\"{env_value}\" disagrees with claim build.full.trunk '{value}'"
        )


# The slots are written in the templates (tools/render.py TEMPLATES); the public README and cards are their renders.
PANEL_FILES = ("card/templates/README.md", "card/templates/MODEL_CARD.md", "card/templates/QUALITY_CARD.md")
PRIORITIES = {"must", "stretch", "later"}


def check_rendered(root: Path, files: list[Path], errors: list[str]) -> None:
    """Each public page equals a fresh render of its template, and no other page holds a slot."""
    sys.path.insert(0, str(root / "tools"))  # check_kit runs with -I, which leaves its own folder off the path
    try:
        import render
    finally:
        sys.path.pop(0)
    try:
        pages = render.render_templates(root, render.load_claims(root / "card" / "claims.json"))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"render: cannot render the templates ({exc})")
        return
    for pub in render.drifted(root, pages):
        errors.append(f"{pub} is missing or differs from a fresh render of its template; run python3 tools/render.py --all")
    templates = set(render.TEMPLATES)
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".md" and rel not in templates and SLOT_RE.search(path.read_text(encoding="utf-8")):
            errors.append(f"{rel} holds a claim slot; slots belong in a template listed in tools/render.py TEMPLATES")


def check_tiers(root: Path, errors: list[str]) -> None:
    """must = the two-build panel: shown on the README or cards and in the fill map; later = never shown."""
    try:
        claims = json.loads((root / "card" / "claims.json").read_text(encoding="utf-8"))["claims"]
        fill = json.loads((root / "card" / "fill-map.json").read_text(encoding="utf-8"))["claims"]
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"tiers: cannot read claims or fill map ({exc})")
        return
    prio = {c.get("id"): c.get("priority") for c in claims}
    for cid, p in prio.items():
        if p not in PRIORITIES:
            errors.append(f"card/claims.json: claim '{cid}' has priority {p!r}, not one of {sorted(PRIORITIES)}")
    shown: set[str] = set()
    for rel in PANEL_FILES:
        path = root / rel
        if path.is_file():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in SLOT_RE.finditer(line):
                    shown.add(match.group(1))
                    if prio.get(match.group(1)) == "later":
                        errors.append(f"{rel}:{lineno}: claim '{match.group(1)}' is tier 'later' and belongs off the cards")
    mapped = {e.get("id") for e in fill}
    for cid, p in prio.items():
        if p == "must" and cid not in shown:
            errors.append(f"must claim '{cid}' is not shown on the README or the cards")
        if p == "must" and cid not in mapped:
            errors.append(f"must claim '{cid}' has no entry in card/fill-map.json")
    for cid in mapped - set(prio):
        errors.append(f"card/fill-map.json: unknown claim '{cid}'")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    files = iter_kit_files(root)
    errors: list[str] = []

    check_required_files(root, errors)
    check_forbidden_strings(root, files, errors)
    check_patches(root, errors)
    check_scripts_and_python(root, files, errors)
    ledger_summary = check_ledger_and_slots(root, files, errors)
    check_trunk_pick(root, errors)
    check_tiers(root, errors)
    check_rendered(root, files, errors)

    print(ledger_summary)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print("check_kit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
