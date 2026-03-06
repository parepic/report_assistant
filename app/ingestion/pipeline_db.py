#!/usr/bin/env python3
"""
Pipeline 1: preprocess -> save_postgresql
"""

from app.ingestion.preprocess import main as preprocess_main
from app.ingestion.save_postgresql import main as save_sql_main
from app.utils.load_utils import load_global_config


def main(on_existing: str = "prompt") -> None:
    """
    Run the preprocessing + PostgreSQL ingestion pipeline for one report.

    Args:
        on_existing: Duplicate handling policy for an existing document ID in
            PostgreSQL. One of "prompt", "skip", or "delete".
    """
    config = load_global_config()
    print("Starting preprocessing...")
    preprocess_main(config)
    print("Preprocessing completed. Saving to PostgreSQL...")
    save_sql_main(config, on_existing=on_existing)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
