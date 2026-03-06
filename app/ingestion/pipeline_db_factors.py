#!/usr/bin/env python3
"""
Pipeline 3: chunk -> save_qdrant (yoy)
"""

from app.ingestion.chunking.chunk import main as chunk_main
from app.utils.load_utils import load_global_config
from app.ingestion.save_postgresql_factors import main as save_postgresql_main_factors

def main(on_existing: str = "prompt") -> None:
    """
    Run the YoY chunking + Qdrant ingestion pipeline for one report.

    Args:
        on_existing: Duplicate handling policy for existing vectors in Qdrant.
            One of "prompt", "skip", or "delete".
    """
    config = load_global_config()
    print("Starting chunking (factor)...")
    chunk_main(config, mode="factor")
    print("Chunking completed")
    save_postgresql_main_factors(config, on_existing=on_existing)    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
