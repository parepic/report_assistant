#!/usr/bin/env python3
"""
Pipeline 2: chunk -> save_qdrant (chatbot)
"""

from app.ingestion.chunking.chunk import main as chunk_main
from app.ingestion.save_qdrant import main as embed_main
from app.utils.load_utils import load_global_config


def main(on_existing: str = "prompt") -> None:
    """
    Run the chatbot chunking + Qdrant ingestion pipeline for one report.

    Args:
        on_existing: Duplicate handling policy for existing vectors in Qdrant.
            One of "prompt", "skip", or "delete".
    """
    config = load_global_config()
    print("Starting chunking (chatbot)...")
    chunk_main(config, mode="chatbot")
    print("Chunking completed. Starting embedding (chatbot)...")
    embed_main(config, mode="chatbot", on_existing=on_existing)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
