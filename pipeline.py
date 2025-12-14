#!/usr/bin/env python3
"""
Pipeline script to run chunking and embedding sequentially.
"""

from report_assistant.chunk import main as chunk_main
from report_assistant.embed import main as embed_main
from report_assistant.llm import main as llm_main


def main():
    print("Starting chunking...")
    chunk_main()
    print("Chunking completed. Starting embedding...")
    embed_main()
    print("LLM conversation starting...")
    llm_main()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()