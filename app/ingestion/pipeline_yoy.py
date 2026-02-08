#!/usr/bin/env python3
"""
YOY pipeline: preprocess -> save_sql -> chunk -> embed
"""

from app.ingestion.preprocess import main as preprocess_main
from app.ingestion.save_postgresql import main as save_sql_main
from app.ingestion.chunking.chunk import main as chunk_main
from app.ingestion.save_qdrant import main as embed_main
from app.utils.load_utils import load_global_config


def main():

    import argparse
    parser = argparse.ArgumentParser(description="YOY pipeline: preprocess, save to SQL, chunk, and embed stages.")
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing stage only')
    parser.add_argument('--save_sql', action='store_true', help='Run save to SQL stage only')
    parser.add_argument('--chunk', action='store_true', help='Run chunking stage only')
    parser.add_argument('--embed', action='store_true', help='Run embedding stage only')

    args = parser.parse_args()

    config = load_global_config()

    # If no flags are set, run all stages
    if not (args.preprocess or args.save_sql or args.chunk or args.embed):
        print("Starting preprocessing...")
        preprocess_main(config)
        print("Preprocessing completed. Saving to SQL...")
        save_sql_main(config)
        print("SQL save completed. Starting chunking...")
        chunk_main(config, mode="yoy")
        print("Chunking completed. Starting embedding...")
        embed_main(config, mode="yoy")
        print("Embedding completed.")
        return


    if args.preprocess:
        print("Starting preprocessing...")
        preprocess_main(config)
        print("Preprocessing completed.")
    if args.save_sql:
        print("Starting save to SQL...")
        save_sql_main(config)
        print("SQL save completed.")
    if args.chunk:
        print("Starting chunking...")
        chunk_main(config, mode="yoy")
        print("Chunking completed.")
    if args.embed:
        print("Starting embedding...")
        embed_main(config, mode="yoy")
        print("Embedding completed.")

if __name__ == "__main__":
    main()
