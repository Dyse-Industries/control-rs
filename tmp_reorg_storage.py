#!/usr/bin/env python3
"""Restore storage.rs PascalCase types to domain order."""

from __future__ import annotations

import re
import sys
from pathlib import Path

KIND_RE = re.compile(
    r"""^
    (?:pub(?:\([^)]*\))?\s+)?
    (?:unsafe\s+)?
    (?:const\s+)?
    (?:async\s+)?
    (use|type|enum|struct|union|trait|impl|fn|mod|macro_rules)
    \b
    """,
    re.VERBOSE,
)
NAME_RE = re.compile(r"\b(?:enum|struct|union|trait|type)\s+(\w+)")

PASCAL = {"type", "enum", "struct", "union", "trait"}
GROUP = {
    "use": "use",
    "type": "pascal",
    "enum": "pascal",
    "struct": "pascal",
    "union": "pascal",
    "trait": "pascal",
    "impl": "impls",
    "fn": "fns",
}

# Original domain order inside the types section.
PASCAL_ORDER = [
    "ColMajorArray",
    "RowMajorArray",
    "ViewMarker",
    "ViewMarkerMut",
    "StorageMarker",
    "SlicePair",
    "UninitArray",
    "MatrixLayout",
    "LayoutMarker",
    "ColMajor",
    "RowMajor",
    "UpLo",
    "Diag",
    "Side",
    "Trans",
    "DenseStorage",
    "DenseStorageMut",
    "ContiguousStorage",
    "ContiguousStorageMut",
    "Storage",
    "StorageMut",
    "StorageInit",
    "ArrayStorage",
    "RowArrayStorage",
    "StorageView",
    "StorageViewMut",
    "StaticStorageView",
    "StaticStorageViewMut",
    "PackedStorage",
    "PackedStorageMut",
    "DiagonalStorage",
    "SymmetricPackedStorage",
    "HermitianPackedStorage",
    "TriangularPackedStorage",
    "DiagonalView",
    "DiagonalViewMut",
    "SymmetricPackedView",
    "SymmetricPackedViewMut",
    "HermitianPackedView",
    "HermitianPackedViewMut",
    "TriangularPackedView",
    "TriangularPackedViewMut",
    "SparseStorage",
    "SparseStorageMut",
    "CsrStorage",
    "CscStorage",
    "SparseVectorStorage",
    "ArrayCooStorage",
    "ArrayCsrStorage",
    "ArrayCscStorage",
    "ArraySparseVector",
    "ViewSparseVector",
    "ToDenseStorage",
    "ToCsrStorage",
    "ToCscStorage",
    "PivotStorage",
]


def split_header(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if (
            stripped.startswith("//!")
            or stripped.startswith("#![")
            or stripped in ("", "\n")
        ):
            i += 1
            continue
        break
    while i > 0 and lines[i - 1].strip() == "":
        i -= 1
    return "".join(lines[:i]).rstrip() + "\n", "".join(lines[i:])


def skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i] in " \t\r\n":
        i += 1
    return i


def scan_item(s: str, start: int) -> int:
    n = len(s)
    i = start
    brace = bracket = paren = 0
    in_line_comment = False
    in_block_comment = 0
    in_string = False
    in_char = False
    string_raw_hashes = None
    escape = False
    seen_body_brace = False

    while i < n:
        c = s[i]
        nxt = s[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if c == "/" and nxt == "*":
                in_block_comment += 1
                i += 2
                continue
            if c == "*" and nxt == "/":
                in_block_comment -= 1
                i += 2
                continue
            i += 1
            continue
        if string_raw_hashes is not None:
            if c == '"':
                hashes = 0
                j = i + 1
                while j < n and s[j] == "#":
                    hashes += 1
                    j += 1
                if hashes >= string_raw_hashes:
                    i = j
                    string_raw_hashes = None
                    continue
            i += 1
            continue
        if in_string:
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\":
                escape = True
                i += 1
                continue
            if c == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\":
                escape = True
                i += 1
                continue
            if c == "'":
                in_char = False
            i += 1
            continue
        if c in "brc" and i + 1 < n:
            j = i
            if s[j] in "bc":
                j += 1
            if j < n and s[j] == "r":
                j += 1
                hashes = 0
                while j < n and s[j] == "#":
                    hashes += 1
                    j += 1
                if j < n and s[j] == '"':
                    string_raw_hashes = hashes
                    i = j + 1
                    continue
        if c == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block_comment = 1
            i += 2
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "'" and not in_char:
            if nxt.isalpha() or nxt == "_":
                i += 1
                while i < n and (s[i].isalnum() or s[i] == "_"):
                    i += 1
                continue
            in_char = True
            i += 1
            continue
        if c == "{":
            brace += 1
            seen_body_brace = True
            i += 1
            continue
        if c == "}":
            brace -= 1
            i += 1
            if brace == 0 and bracket == 0 and paren == 0 and seen_body_brace:
                j = i
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j < n and s[j] == ";":
                    return j + 1
                return i
            continue
        if c == "[":
            bracket += 1
            i += 1
            continue
        if c == "]":
            bracket -= 1
            i += 1
            continue
        if c == "(":
            paren += 1
            i += 1
            continue
        if c == ")":
            paren -= 1
            i += 1
            continue
        if c == ";" and brace == 0 and bracket == 0 and paren == 0:
            return i + 1
        i += 1
    return n


def classify(text: str) -> str:
    code = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#["):
            continue
        code.append(stripped)
    blob = " ".join(code)
    m = KIND_RE.match(blob)
    if not m:
        raise ValueError(blob[:120])
    return m.group(1)


def item_name(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#["):
            continue
        m = NAME_RE.search(stripped)
        if m:
            return m.group(1)
        break
    return None


def split_items(body: str) -> list[str]:
    items = []
    i = skip_ws(body, 0)
    n = len(body)
    while i < n:
        end = scan_item(body, i)
        item = body[i:end].strip("\n")
        if item.strip():
            items.append(item)
        i = skip_ws(body, end)
    return items


def main() -> int:
    path = Path("src/math/storage.rs")
    original = path.read_text()
    header, body = split_header(original)
    raw = split_items(body)
    classified = [(classify(t), t) for t in raw]

    groups = {"use": [], "pascal": [], "impls": [], "fns": []}
    for kind, text in classified:
        groups[GROUP[kind]].append(text)

    by_name = {}
    unnamed = []
    for text in groups["pascal"]:
        name = item_name(text)
        if name is None:
            unnamed.append(text)
            continue
        if name in by_name:
            print(f"duplicate name {name}", file=sys.stderr)
            return 1
        by_name[name] = text

    ordered = []
    seen = set()
    for name in PASCAL_ORDER:
        if name in by_name:
            ordered.append(by_name[name])
            seen.add(name)
    leftover = [n for n in by_name if n not in seen]
    if leftover:
        print("unordered pascal items:", leftover, file=sys.stderr)
        return 1
    ordered.extend(unnamed)
    groups["pascal"] = ordered

    def compact(key: str, items: list[str]) -> str:
        if key in ("use",) or (
            key == "pascal" and False
        ):
            return "\n".join(items)
        # Keep type aliases tight at the top of pascal.
        if key == "pascal":
            aliases, rest = [], []
            for item in items:
                if classify(item) == "type":
                    aliases.append(item)
                else:
                    rest.append(item)
            block = "\n".join(aliases)
            if rest:
                block = block + "\n\n" + "\n\n".join(rest)
            return block
        return "\n\n".join(items)

    parts = [header.rstrip(), ""]
    for key in ("use", "pascal", "impls", "fns"):
        if groups[key]:
            parts.append(compact(key, groups[key]))
            parts.append("")
    new = "\n".join(parts).rstrip() + "\n"

    orig_set = sorted(re.sub(r"\s+", " ", t).strip() for _, t in classified)
    new_set = sorted(
        re.sub(r"\s+", " ", t).strip() for t in split_items(split_header(new)[1])
    )
    if orig_set != new_set:
        print("item set mismatch", file=sys.stderr)
        return 1

    path.write_text(new)
    print(f"wrote {path} ({len(groups['pascal'])} types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
