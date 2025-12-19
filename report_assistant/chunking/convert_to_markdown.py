"""
Utility to convert Word documents (.docx) to Markdown.

Tables are rendered as Markdown tables and basic headings/bullets are preserved
using style names from the source document. This keeps content LLM-friendly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
import pypandoc


# Convert using pypandoc
def convert_to_markdown_pypandoc(
    input_path: Path,
    # output_path: Path
) -> str:
    """
    Convert a .docx file to Markdown using pypandoc.
    Returns the Markdown text.
    """
    return pypandoc.convert_file(
        str(input_path),
        "markdown",
        # outputfile=str(output_path),
        extra_args=["--standalone",
                    # f"--extract-media={output_path.parent / 'media'}"
                    ],
    )


# Convert using python-docx

def docx_to_markdown(input_path: Path) -> str:
    """
    Convert a .docx file to Markdown text.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    doc = Document(input_path)
    md_parts: List[str] = []

    for block in iter_blocks(doc):
        if isinstance(block, Paragraph):
            md = paragraph_to_md(block)
        else:
            md = table_to_md(block)
        if md:
            md_parts.append(md)

    return "\n\n".join(md_parts)





def iter_blocks(doc: Document) -> Iterable[Paragraph | Table]:
    """
    Yield paragraphs and tables in document order.
    """
    for child in doc.element.body:
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def paragraph_to_md(paragraph: Paragraph) -> str:
    """
    Convert a paragraph to a Markdown string with simple style handling.
    """
    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = (paragraph.style.name or "").lower()

    # Headings: map "Heading 1" -> "#", etc.
    if style_name.startswith("heading"):
        parts = style_name.split()
        level = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        level = max(1, min(level, 6))
        return f"{'#' * level} {text}"

    # Bulleted / list paragraphs
    if "bullet" in style_name or "list" in style_name:
        return f"- {text}"

    return text


def table_to_md(table: Table) -> str:
    """
    Convert a Word table to Markdown table syntax.
    """
    rows: List[List[str]] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append(cells)

    if not rows:
        return ""

    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    md_lines = [fmt_row(header)]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        md_lines.append(fmt_row(r))

    return "\n".join(md_lines)