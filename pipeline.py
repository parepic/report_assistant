#!/usr/bin/env python3
"""
Pipeline script to run chunking and embedding sequentially.
"""

from report_assistant.preprocess import main as preprocess_main
from report_assistant.chunking.chunk import main as chunk_main
from report_assistant.embed import main as embed_main
from report_assistant.llm import main as llm_main
from tests.test import main as test_main
from report_assistant.save_sql import main as save_sql_main


from report_assistant.utils.load_utils import load_global_config


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pipeline for chunking, embedding, and LLM stages.")
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing stage only')
    parser.add_argument('--chunk', action='store_true', help='Run chunking stage only')
    parser.add_argument('--save_sql', action='store_true', help='Run save to SQL stage only')
    parser.add_argument('--embed', action='store_true', help='Run embedding stage only')
    parser.add_argument('--llm', action='store_true', help='Run LLM stage only')
    parser.add_argument('--test', action='store_true', help='Run test stage only')

    args = parser.parse_args()

    config = load_global_config()

    # If no flags are set, run all stages
    if not (args.preprocess or args.chunk or args.save_sql or args.embed or args.llm or args.test):
        print("Starting preprocessing...")
        preprocess_main(config)
        chunk_main(config)
        print("Chunking completed. Saving to SQL...")
        save_sql_main(config)
        print("SQL save completed. Starting embedding...")
        embed_main(config)
        print("Embedding completed. Comparing expected answers with LLM answers...")
        test_main(config)
        print("Pipeline completed successfully.")
        return

    if args.preprocess:
        print("Starting preprocessing...")
        preprocess_main(config)
        print("Preprocessing completed.")
    if args.chunk:
        print("Starting chunking...")
        chunk_main(config)
        print("Chunking completed.")
    if args.save_sql:
        print("Starting save to SQL...")
        save_sql_main(config)
        print("SQL save completed.")
    if args.embed:
        print("Starting embedding...")
        embed_main(config)
        print("Embedding completed.")
    if args.llm:
        print("LLM conversation starting...")
        llm_main(config)
        print("LLM stage completed.")
    if args.test:
        print("Test stage starting...")
        test_main(config)
        print("Test stage completed.")


if __name__ == "__main__":
    main()


    