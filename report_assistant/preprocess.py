"""
Document preprocessing module.

Handles conversion of various document formats to Markdown and cleanup operations.
"""
from __future__ import annotations

from pathlib import Path
import warnings

from docx import Document

from report_assistant.chunking.convert_to_markdown import (
    convert_to_markdown_pypandoc
)
from report_assistant.data_classes import GlobalConfig
from report_assistant.utils.load_utils import get_index_path, load_document_entry

import warnings

from typing import List
import re

from docx import Document
import pypandoc


ROW_END_TOKEN = "<ROW_END>"
TABLE_END_TOKEN = "<TABLE_END>."


def convert_to_markdown_pypandoc(
    input_path: Path,
    # output_path: Path
) -> str:
    """
    Convert a .docx file to Markdown using pypandoc.
    Returns the Markdown text.
    """
    warnings.warn(
        "Deprecated: use docx_to_markdown instead.",
        UserWarning,
        stacklevel=2,
    )

    return pypandoc.convert_file(
        str(input_path),
        "markdown",
        # outputfile=str(output_path),
        extra_args=["--standalone",
                    "--wrap=none",
                    # f"--extract-media={output_path.parent / 'media'}"
                    ],
    )

def clean_markdown_text(md_text: str) -> str:
    """
    Clean up some common pypandoc artifacts in the Markdown text.
    """
    # Remove excessive newlines
    md_text = remove_excessive_newlines(md_text)
    md_text = remove_text_before_item_1a(md_text)
    md_text = remove_text_after_item_section(md_text)
    md_text = remove_unwanted_lines(md_text)
    md_text = remove_gt_markers(md_text)
    md_text = merge_paragraphs_without_terminal_punct(md_text)

    return md_text

def remove_excessive_newlines(md_text: str) -> str:
    lines = md_text.splitlines()
    cleaned_lines = []
    previous_blank = False
    for line in lines:
        if line.strip() == "":
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False
    return "\n".join(cleaned_lines)


def remove_text_before_item_1a(md_text: str) -> str:
    """
    Remove all text until encountering "Item 1A." (inclusive of that line).
    """
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        if "item 1a." in line.lower():
            return "\n".join(lines[i+1:])
    return md_text


def remove_text_after_item_section(md_text: str) -> str:
    """
    Remove everything after encountering a line that, after removing '#' characters,
    becomes "item 1" or "item 2" etc. (case-insensitive).
    """
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        # Remove all '#' characters and strip whitespace
        cleaned_line = line.replace('#', '').strip().lower()
        # Check if it matches pattern "item <number>"
        if re.match(r'^item\s+\d+', cleaned_line):
            # Return everything before this line (exclusive)
            return "\n".join(lines[:i])
    return md_text


def remove_unwanted_lines(md_text: str) -> str:
    """
    Remove lines that are only a single number, start with '![]', or start with '[Table of Contents]'.
    Leading whitespace is ignored for the prefix checks.
    """
    lines = md_text.splitlines()
    kept: List[str] = []

    for line in lines:
        stripped = line.strip()
        lstripped = line.lstrip()

        if re.fullmatch(r"\d+", stripped):
            continue
        if lstripped.startswith("![]"):
            continue
        if lstripped.startswith("Table of Contents"):
            continue

        kept.append(line)

    return "\n".join(kept)


def remove_gt_markers(md_text: str) -> str:
    """
    Remove blockquote-style markers: leading '>' (with optional spaces) and
    the exact substring " >" wherever it appears.
    This avoids carrying over blockquote arrows into cleaned Markdown.
    """
    lines = md_text.splitlines()
    out: List[str] = []
    for line in lines:
        # Remove leading '>' with optional whitespace after it
        stripped_lead = re.sub(r"^\s*>\s*", "", line)
        # Remove the exact substring space+greater-than
        cleaned = stripped_lead.replace(" >", "")
        out.append(cleaned)
    return "\n".join(out)


def merge_paragraphs_without_terminal_punct(md_text: str) -> str:
    """
    Merge consecutive paragraphs when the previous paragraph does not end with
    '.', ',' or ';'. Paragraphs are separated by one or more blank lines.
    Paragraphs starting with '#' are not merged.
    """
    text = md_text.strip("\n")
    if not text:
        return md_text

    paragraphs = re.split(r"\n\s*\n", text)
    merged: List[str] = []
    buffer = ""

    for paragraph in paragraphs:
        p = paragraph.strip()
        if not p:
            continue

        if p.startswith("#"):
            if buffer:
                merged.append(buffer)
            merged.append(p)
            buffer = ""
            continue

        if not buffer:
            buffer = p
            continue

        if buffer.lstrip().startswith("#"):
            merged.append(buffer)
            buffer = p
            continue

        if not buffer.rstrip().endswith((".")):
            buffer = buffer.rstrip() + "\n" + p.lstrip()
        else:
            merged.append(buffer)
            buffer = p

    if buffer:
        merged.append(buffer)

    suffix = "\n" if md_text.endswith("\n") else ""
    return "\n\n".join(merged) + suffix




def load_text_from_docx(path: Path) -> str:
    """
    Read all text from a .docx file into a single string.
    """
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def preprocess_document(file_path: Path, output_path: Path, file_format: str = "docx") -> str:
    """
    Convert a document to cleaned Markdown format.
    
    Args:
        file_path: Path to the source document
        output_path: Path where the markdown file will be saved
        file_format: Format of the source document (default: "docx")
    
    Returns:
        The cleaned markdown text
    """
    if file_format != "docx":
        raise ValueError(f"Unsupported document format: {file_format}")
    
    # Convert to markdown
    markdown_text = convert_to_markdown_pypandoc(file_path)
    
    # Clean the markdown
    markdown_text = clean_markdown_text(markdown_text)
    
    # Save to file
    output_path.write_text(markdown_text, encoding="utf-8")
    
    return markdown_text


def main(config: GlobalConfig) -> None:
    """
    Main preprocessing function that processes documents based on config.
    """
    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    file_path = entry.source_file_path
    
    # Prepare output path
    markdown_path = entry.text_dir / f"{entry.doc_id}.md"
    
    preprocess_document(
        file_path=file_path,
        output_path=markdown_path,
        file_format=entry.source_format
    )
    
    print(
        f"Preprocessed {config.report_id}. "
        f"Converted to markdown and saved to {markdown_path}"
    )


if __name__ == "__main__":
    from report_assistant.utils.load_utils import load_global_config
    config = load_global_config()
    main(config)
