#!/usr/bin/env python3
"""
Pipeline script to run chunking and embedding sequentially.
"""

from report_assistant.chunking.chunk import main as chunk_main
from report_assistant.embed import main as embed_main
from report_assistant.llm import main as llm_main
from report_assistant.utils.load_utils import load_global_config


def main():
    config = load_global_config()

    print("Starting chunking...")
    chunk_main(config)
    print("Chunking completed. Starting embedding...")
    embed_main(config)
    print("LLM conversation starting...")
    llm_main(config)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()


    