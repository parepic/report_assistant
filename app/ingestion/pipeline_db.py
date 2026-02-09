#!/usr/bin/env python3
"""
Pipeline 1: preprocess -> save_postgresql
"""

from app.ingestion.preprocess import main as preprocess_main
from app.ingestion.save_postgresql import main as save_sql_main
from app.utils.load_utils import load_global_config


def main() -> None:
    config = load_global_config()
    print("Starting preprocessing...")
    preprocess_main(config)
    print("Preprocessing completed. Saving to PostgreSQL...")
    save_sql_main(config)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
