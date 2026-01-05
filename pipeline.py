#!/usr/bin/env python3
"""
Pipeline script to run chunking and embedding sequentially.
"""

from report_assistant.chunking.chunk import main as chunk_main
from report_assistant.embed import main as embed_main
from report_assistant.llm import main as llm_main
from report_assistant.test import main as test_main

from report_assistant.utils.load_utils import load_global_config


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Pipeline for chunking, embedding, and LLM stages.")
    parser.add_argument('--chunk', action='store_true', help='Run chunking stage only')
    parser.add_argument('--embed', action='store_true', help='Run embedding stage only')
    parser.add_argument('--llm', action='store_true', help='Run LLM stage only')
    parser.add_argument('--test', action='store_true', help='Run test stage only')

    args = parser.parse_args()

    config = load_global_config()

    # If no flags are set, run all stages
    if not (args.chunk or args.embed or args.llm):
        print("Starting chunking...")
        chunk_main(config)
        print("Chunking completed. Starting embedding...")
        embed_main(config)
        print("Comparing expected answers with LLM answers...")
        test_main(config)
        print("Pipeline completed successfully.")
        return

    if args.chunk:
        print("Starting chunking...")
        chunk_main(config)
        print("Chunking completed.")
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


    