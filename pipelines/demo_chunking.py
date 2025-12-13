"""
Quick demonstration of how blocks (paragraph-like units) are assembled into chunks.

Run:
    python -m pipelines.demo_chunking
"""
from __future__ import annotations

from pipelines.text_pipeline import Chunk, TextBlock, chunk_blocks


def make_sample_blocks() -> list[TextBlock]:
    return [
        TextBlock(page=1, section="Intro", text="Microsoft delivered strong results in FY2024."),
        TextBlock(page=1, section="Intro", text="Revenue grew double digits across cloud and productivity."),
        TextBlock(page=2, section="Risk Factors", text="Key risks include macro uncertainty and FX headwinds."),
        TextBlock(page=2, section="Risk Factors", text="Security remains a top priority across all products."),
        TextBlock(page=3, section="Outlook", text="Management expects continued AI-driven demand."),
    ]


def describe_chunk(chunk: Chunk) -> str:
    return (
        f"{chunk.chunk_id} | pages={chunk.page_span} | section={chunk.section or '-'} | "
        f"tokens={len(chunk.text.split())} | text='{chunk.text}'"
    )


def main() -> None:
    blocks = make_sample_blocks()
    print("Blocks:")
    for i, b in enumerate(blocks, start=1):
        print(f"  block_{i:02d} | page={b.page} | section={b.section} | text='{b.text}'")

    print("\nChunks (chunk_size_tokens=12, overlap_tokens=4):")
    chunks = chunk_blocks(blocks, chunk_size_tokens=12, overlap_tokens=4)
    for c in chunks:
        print(" ", describe_chunk(c))


if __name__ == "__main__":
    main()
