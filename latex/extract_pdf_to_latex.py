#!/usr/bin/env python3
"""Extract a text-based PDF into raw text and a reconstructed LaTeX file.

This is a best-effort extractor for Chrome/Edge generated PDFs that encode
text via ToUnicode CMaps. It is intentionally lightweight and avoids external
dependencies because the current environment does not provide pdf parsing
libraries or command-line tools such as pdftotext.
"""

from __future__ import annotations

import argparse
import re
import zlib
from dataclasses import dataclass
from pathlib import Path


OBJ_PAT = re.compile(rb"(?m)^(\d+)\s+(\d+)\s+obj\b")
REF_PAT = re.compile(rb"(\d+)\s+(\d+)\s+R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Input PDF. Defaults to the largest PDF in the current directory.",
    )
    parser.add_argument(
        "--raw-output",
        default="feasibility_validation_from_pdf_raw.txt",
        help="Path to the extracted raw text output.",
    )
    parser.add_argument(
        "--tex-output",
        default="feasibility_validation_from_pdf.tex",
        help="Path to the reconstructed LaTeX output.",
    )
    return parser.parse_args()


def pick_default_pdf() -> Path:
    pdfs = sorted(Path(".").glob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)
    if not pdfs:
        raise FileNotFoundError("No PDF file found in the current directory.")
    return pdfs[0]


def parse_objects(pdf_bytes: bytes) -> dict[tuple[int, int], bytes]:
    objects: dict[tuple[int, int], bytes] = {}
    matches = list(OBJ_PAT.finditer(pdf_bytes))
    for i, match in enumerate(matches):
        obj_num = int(match.group(1))
        gen_num = int(match.group(2))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pdf_bytes)
        body = pdf_bytes[start:end]
        endobj = body.find(b"endobj")
        if endobj != -1:
            body = body[:endobj]
        objects[(obj_num, gen_num)] = body.strip()
    return objects


def split_stream(body: bytes) -> tuple[bytes, bytes | None]:
    if b"stream" not in body:
        return body, None
    head, rest = body.split(b"stream", 1)
    if b"endstream" not in rest:
        return head, None
    stream, _ = rest.split(b"endstream", 1)
    stream = stream.lstrip(b"\r\n").rstrip(b"\r\n")
    if b"/FlateDecode" in head:
        try:
            return head, zlib.decompress(stream)
        except Exception:
            return head, None
    return head, stream


def extract_simple_dict(body: bytes, name: bytes) -> dict[str, int]:
    match = re.search(rb"/" + name + rb"\s*<<(.*?)>>", body, re.S)
    if not match:
        return {}
    return {
        key.decode("latin1"): int(ref)
        for key, ref in re.findall(rb"/([A-Za-z0-9]+)\s+(\d+)\s+\d+\s+R", match.group(1))
    }


def parse_cmap(cmap_stream: bytes) -> dict[bytes, str]:
    text = cmap_stream.decode("latin1", errors="ignore")
    lines = [line.strip() for line in text.splitlines()]
    mapping: dict[bytes, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.endswith("beginbfchar"):
            count = int(line.split()[0])
            for offset in range(1, count + 1):
                parts = re.findall(r"<([^>]*)>", lines[idx + offset])
                if len(parts) >= 2:
                    mapping[bytes.fromhex(parts[0])] = bytes.fromhex(parts[1]).decode(
                        "utf-16-be", errors="ignore"
                    )
            idx += count + 1
            continue
        if line.endswith("beginbfrange"):
            count = int(line.split()[0])
            for offset in range(1, count + 1):
                raw_line = lines[idx + offset]
                parts = re.findall(r"<([^>]*)>", raw_line)
                if len(parts) < 3:
                    continue
                src_start = int(parts[0], 16)
                src_end = int(parts[1], 16)
                src_len = len(parts[0]) // 2
                if "[" in raw_line:
                    dsts = re.findall(r"<([^>]*)>", raw_line.split("[", 1)[1])
                    for enum_idx, code in enumerate(range(src_start, src_end + 1)):
                        if enum_idx < len(dsts):
                            mapping[code.to_bytes(src_len, "big")] = bytes.fromhex(
                                dsts[enum_idx]
                            ).decode("utf-16-be", errors="ignore")
                else:
                    dst_start = int(parts[2], 16)
                    dst_len = max(2, len(parts[2]) // 2)
                    for enum_idx, code in enumerate(range(src_start, src_end + 1)):
                        mapping[code.to_bytes(src_len, "big")] = (
                            dst_start + enum_idx
                        ).to_bytes(dst_len, "big").decode("utf-16-be", errors="ignore")
            idx += count + 1
            continue
        idx += 1
    return mapping


def build_font_cmaps(objects: dict[tuple[int, int], bytes]) -> dict[int, dict[bytes, str]]:
    font_cmaps: dict[int, dict[bytes, str]] = {}
    for (obj_num, gen_num), body in objects.items():
        if gen_num != 0:
            continue
        match = re.search(rb"/ToUnicode\s+(\d+)\s+(\d+)\s+R", body)
        if not match:
            continue
        cmap_ref = (int(match.group(1)), int(match.group(2)))
        cmap_body = objects.get(cmap_ref)
        if not cmap_body:
            continue
        _, stream = split_stream(cmap_body)
        if stream is None:
            continue
        font_cmaps[obj_num] = parse_cmap(stream)
    return font_cmaps


def find_pages_root(objects: dict[tuple[int, int], bytes]) -> int:
    for (obj_num, gen_num), body in objects.items():
        if gen_num != 0:
            continue
        if b"/Type /Catalog" in body:
            match = re.search(rb"/Pages\s+(\d+)\s+\d+\s+R", body)
            if match:
                return int(match.group(1))
    raise RuntimeError("Could not find /Catalog /Pages root.")


def build_page_tree(objects: dict[tuple[int, int], bytes], root_pages: int) -> list[int]:
    page_children: dict[int, list[int]] = {}
    leaf_pages: set[int] = set()
    for (obj_num, gen_num), body in objects.items():
        if gen_num != 0:
            continue
        if b"/Type /Pages" in body:
            match = re.search(rb"/Kids\s*\[(.*?)\]", body, re.S)
            if match:
                page_children[obj_num] = [
                    int(ref) for ref, _ in REF_PAT.findall(match.group(1))
                ]
        elif b"/Type /Page" in body:
            leaf_pages.add(obj_num)

    ordered: list[int] = []

    def walk(node: int) -> None:
        if node in leaf_pages:
            ordered.append(node)
            return
        for child in page_children.get(node, []):
            walk(child)

    walk(root_pages)
    return ordered


def decode_hex_string(hex_string: str, cmap: dict[bytes, str]) -> str:
    raw = bytes.fromhex(hex_string)
    lengths = sorted({len(key) for key in cmap.keys()}, reverse=True) or [2, 1]
    decoded: list[str] = []
    idx = 0
    while idx < len(raw):
        matched = False
        for length in lengths:
            chunk = raw[idx : idx + length]
            if chunk in cmap:
                decoded.append(cmap[chunk])
                idx += length
                matched = True
                break
        if matched:
            continue
        try:
            decoded.append(raw[idx : idx + 1].decode("latin1"))
        except Exception:
            pass
        idx += 1
    return "".join(decoded)


def decode_literal_string(literal: str) -> str:
    out: list[str] = []
    idx = 0
    while idx < len(literal):
        ch = literal[idx]
        if ch != "\\":
            out.append(ch)
            idx += 1
            continue
        idx += 1
        if idx >= len(literal):
            break
        esc = literal[idx]
        if esc in "()\\":
            out.append(esc)
        elif esc == "n":
            out.append("\n")
        elif esc == "r":
            out.append("\r")
        elif esc == "t":
            out.append("\t")
        elif esc == "b":
            out.append("\b")
        elif esc == "f":
            out.append("\f")
        elif esc.isdigit():
            oct_digits = esc
            for _ in range(2):
                if idx + 1 < len(literal) and literal[idx + 1].isdigit():
                    idx += 1
                    oct_digits += literal[idx]
                else:
                    break
            out.append(chr(int(oct_digits, 8)))
        else:
            out.append(esc)
        idx += 1
    return "".join(out)


def decode_tj_array(array_text: str, cmap: dict[bytes, str]) -> str:
    parts: list[str] = []
    pattern = re.compile(r"<([0-9A-Fa-f]+)>|\(((?:\\.|[^\\)])*)\)")
    for match in pattern.finditer(array_text):
        if match.group(1):
            parts.append(decode_hex_string(match.group(1), cmap))
        else:
            parts.append(decode_literal_string(match.group(2)))
    return "".join(parts)


def cleanup_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class LineItem:
    text: str
    size: float


TOKEN_PAT = re.compile(
    r"""
    /(?P<font>[A-Za-z0-9]+)\s+(?P<size>[-+]?\d*\.?\d+)\s+Tf |
    (?P<tm>[-+]?\d*\.?\d+(?:\s+[-+]?\d*\.?\d+){5})\s+Tm |
    (?P<tdx>[-+]?\d*\.?\d+)\s+(?P<tdy>[-+]?\d*\.?\d+)\s+Td |
    (?P<tdx2>[-+]?\d*\.?\d+)\s+(?P<tdy2>[-+]?\d*\.?\d+)\s+TD |
    (?P<tstar>T\*) |
    <(?P<hex>[0-9A-Fa-f]+)>\s*Tj |
    \[(?P<tj>.*?)\]\s*TJ |
    \((?P<lit>(?:\\.|[^\\)])*)\)\s*Tj |
    /(?P<xobj>[A-Za-z0-9]+)\s+Do |
    (?P<bt>BT) |
    (?P<et>ET)
    """,
    re.S | re.X,
)


def should_filter_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("Thought for "):
        return True
    if stripped == "PDF":
        return True
    if stripped in {
        "arXiv",
        "NVIDIA Isaac ROS",
        "edge_gaussian_report",
        "edge_gaussian_report.",
        "+",
        "…",
    }:
        return True
    if re.fullmatch(r"\d+", stripped):
        return True
    if stripped.startswith("帮我完整理解这个项目"):
        return True
    return False


def extract_lines_from_stream(
    obj_num: int,
    objects: dict[tuple[int, int], bytes],
    font_cmaps: dict[int, dict[bytes, str]],
    resources: dict[str, dict[str, int]],
    seen_streams: set[int],
) -> list[LineItem]:
    if obj_num in seen_streams:
        return []
    seen_streams.add(obj_num)

    body = objects.get((obj_num, 0))
    if not body:
        return []
    head, stream = split_stream(body)
    if stream is None:
        return []

    stream_text = stream.decode("latin1", errors="ignore")
    current_font_name: str | None = None
    current_size = 12.0
    current_y = 0.0
    current_line: list[str] = []
    lines: list[LineItem] = []

    def flush_line() -> None:
        if not current_line:
            return
        joined = cleanup_text("".join(current_line))
        current_line.clear()
        if joined and not should_filter_line(joined):
            lines.append(LineItem(text=joined, size=current_size))

    for match in TOKEN_PAT.finditer(stream_text):
        if match.group("font"):
            current_font_name = match.group("font")
            current_size = float(match.group("size"))
            continue

        if match.group("tm"):
            values = [float(value) for value in match.group("tm").split()]
            new_y = values[5]
            if current_line and abs(new_y - current_y) > 0.1:
                flush_line()
            current_y = new_y
            continue

        if match.group("tdx") and match.group("tdy"):
            dy = float(match.group("tdy"))
            if current_line and abs(dy) > 0.1:
                flush_line()
            current_y += dy
            continue

        if match.group("tdx2") and match.group("tdy2"):
            dy = float(match.group("tdy2"))
            if current_line and abs(dy) > 0.1:
                flush_line()
            current_y += dy
            continue

        if match.group("tstar"):
            flush_line()
            continue

        if match.group("hex"):
            cmap = font_cmaps.get(resources["fonts"].get(current_font_name or "", -1), {})
            current_line.append(decode_hex_string(match.group("hex"), cmap))
            continue

        if match.group("tj"):
            cmap = font_cmaps.get(resources["fonts"].get(current_font_name or "", -1), {})
            current_line.append(decode_tj_array(match.group("tj"), cmap))
            continue

        if match.group("lit"):
            current_line.append(decode_literal_string(match.group("lit")))
            continue

        if match.group("xobj"):
            flush_line()
            xobj_name = match.group("xobj")
            xobj_num = resources["xobjects"].get(xobj_name)
            if xobj_num is None:
                continue
            xobj_body = objects.get((xobj_num, 0))
            if not xobj_body or b"/Subtype /Form" not in xobj_body:
                continue
            nested_resources = {
                "fonts": dict(resources["fonts"]),
                "xobjects": dict(resources["xobjects"]),
            }
            nested_fonts = extract_simple_dict(xobj_body, b"Font")
            nested_xobjects = extract_simple_dict(xobj_body, b"XObject")
            nested_resources["fonts"].update(nested_fonts)
            nested_resources["xobjects"].update(nested_xobjects)
            lines.extend(
                extract_lines_from_stream(
                    xobj_num, objects, font_cmaps, nested_resources, seen_streams
                )
            )
            continue

        if match.group("bt"):
            flush_line()
            continue

        if match.group("et"):
            flush_line()

    flush_line()
    return lines


def extract_document_lines(
    objects: dict[tuple[int, int], bytes], font_cmaps: dict[int, dict[bytes, str]]
) -> list[tuple[int, list[LineItem]]]:
    root_pages = find_pages_root(objects)
    ordered_pages = build_page_tree(objects, root_pages)
    document_lines: list[tuple[int, list[LineItem]]] = []

    for page_index, page_obj in enumerate(ordered_pages, start=1):
        page_body = objects[(page_obj, 0)]
        resources = {
            "fonts": extract_simple_dict(page_body, b"Font"),
            "xobjects": extract_simple_dict(page_body, b"XObject"),
        }
        contents_match = re.search(rb"/Contents\s*(\[(.*?)\]|(\d+\s+\d+\s+R))", page_body, re.S)
        if not contents_match:
            document_lines.append((page_index, []))
            continue

        content_refs = [
            int(obj_num) for obj_num, _ in REF_PAT.findall(contents_match.group(1))
        ]
        page_lines: list[LineItem] = []
        seen_streams: set[int] = set()
        for ref in content_refs:
            page_lines.extend(
                extract_lines_from_stream(ref, objects, font_cmaps, resources, seen_streams)
            )
        document_lines.append((page_index, page_lines))
    return document_lines


MAIN_HEADING_RE = re.compile(r"^[一二三四五六七八九十百]+、")
SUB_HEADING_RE = re.compile(r"^\d+[）)]")


def merge_page_lines(page_lines: list[LineItem]) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    pending_bullet: str | None = None

    for item in page_lines:
        line = item.text.strip()
        if not line:
            continue
        if MAIN_HEADING_RE.match(line):
            blocks.append(("section", line))
            continue
        if SUB_HEADING_RE.match(line):
            blocks.append(("subsection", line))
            continue
        if re.fullmatch(r"\d+[.．、]?", line):
            pending_bullet = line.rstrip(".．、")
            continue
        if pending_bullet is not None:
            blocks.append(("paragraph", f"{pending_bullet}. {line}"))
            pending_bullet = None
            continue
        if line.endswith("：") and 4 <= len(line) <= 32:
            blocks.append(("paragraph_heading", line[:-1]))
            continue
        blocks.append(("paragraph", line))

    if pending_bullet is not None:
        blocks.append(("paragraph", pending_bullet))
    return blocks


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_raw_text(output_path: Path, document_lines: list[tuple[int, list[LineItem]]]) -> None:
    parts: list[str] = []
    for page_number, lines in document_lines:
        parts.append(f"===== Page {page_number} =====")
        parts.extend(line.text for line in lines)
        parts.append("")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_latex(
    output_path: Path,
    source_pdf: Path,
    document_lines: list[tuple[int, list[LineItem]]],
) -> None:
    title = source_pdf.stem + "（PDF 转写版）"
    tex_parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[UTF8]{ctex}",
        r"\usepackage{geometry}",
        r"\usepackage{hyperref}",
        r"\geometry{a4paper,margin=1in}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        "",
        rf"\title{{{latex_escape(title)}}}",
        r"\author{Automatically reconstructed from PDF}",
        r"\date{}",
        "",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section*{说明}",
        (
            "本文件由原始 PDF 自动抽取并重建为 LaTeX。文本内容以高保真转写为优先，"
            "章节结构、列表、表格、公式与图片位置为近似重建，仍建议人工复核。"
        ),
        "",
    ]

    for page_number, page_lines in document_lines:
        tex_parts.append(f"% Page {page_number}")
        for block_type, text in merge_page_lines(page_lines):
            escaped = latex_escape(text)
            if block_type == "section":
                tex_parts.append(rf"\section{{{escaped}}}")
            elif block_type == "subsection":
                tex_parts.append(rf"\subsection{{{escaped}}}")
            elif block_type == "paragraph_heading":
                tex_parts.append(rf"\paragraph{{{escaped}}}")
            else:
                tex_parts.append(escaped)
                tex_parts.append("")

    tex_parts.extend([r"\end{document}", ""])
    output_path.write_text("\n".join(tex_parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf) if args.pdf else pick_default_pdf()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    objects = parse_objects(pdf_path.read_bytes())
    font_cmaps = build_font_cmaps(objects)
    document_lines = extract_document_lines(objects, font_cmaps)

    raw_output = Path(args.raw_output)
    tex_output = Path(args.tex_output)
    write_raw_text(raw_output, document_lines)
    write_latex(tex_output, pdf_path, document_lines)

    print(f"Input PDF: {pdf_path}")
    print(f"Raw text written to: {raw_output}")
    print(f"LaTeX written to: {tex_output}")
    print(f"Extracted pages: {len(document_lines)}")


if __name__ == "__main__":
    main()
