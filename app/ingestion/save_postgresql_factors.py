from __future__ import annotations

from typing import Dict, List

from app.data_classes import GlobalConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.utils.load_utils import get_index_path, load_document_entry, load_chunks
from app.utils.utils import init_db
from app.models import Document, RiskFactor


def save_factors_to_db(
    factors: List[Dict],
    document_id: str,
    engine,
    on_existing: str = "prompt",
) -> None:
    """
    Save risk-factor chunks for one document into the `risk_factors` table.

    This function inserts one row per factor chunk, linking each row to the
    parent `documents.id` via `document_id`.

    Duplicate handling behavior is controlled by `on_existing` and follows
    the same pattern as `save_postgresql.py`:
    - "prompt": ask whether to delete existing rows for this document and re-insert.
    - "skip": keep existing rows unchanged and skip insertion.
    - "delete": remove existing rows for this document, then insert new rows.

    Args:
        factors: List of chunk dictionaries, expected to contain at least `text`
            and optionally `risk_factor` and `idx`.
        document_id: The parent document identifier (`documents.id`).
        engine: SQLAlchemy engine connected to PostgreSQL.
        on_existing: Duplicate handling policy. One of "prompt", "skip", "delete".

    Returns:
        None.
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        existing_doc = session.query(Document).filter(Document.id == document_id).first()
        if not existing_doc:
            raise ValueError(
                f"Cannot save risk factors: document '{document_id}' is missing in 'documents'. "
                "Run the document save pipeline first."
            )

        existing_factor = (
            session.query(RiskFactor)
            .filter(RiskFactor.document_id == document_id)
            .first()
        )

        if existing_factor:
            print(
                f"\n⚠️  Risk factors for document '{document_id}' already exist in the database."
            )
            if on_existing == "skip":
                print("Skipping insertion. Existing risk factors remain unchanged.")
                return
            if on_existing == "delete":
                print("Deleting existing risk factors and re-inserting...")
                (
                    session.query(RiskFactor)
                    .filter(RiskFactor.document_id == document_id)
                    .delete(synchronize_session=False)
                )
                session.commit()
                print("Existing risk factors deleted.")
            else:
                response = input("Delete and re-insert risk factors? (yes/no): ").strip().lower()
                if response in ("yes", "y"):
                    print("Deleting existing risk factors and re-inserting...")
                    (
                        session.query(RiskFactor)
                        .filter(RiskFactor.document_id == document_id)
                        .delete(synchronize_session=False)
                    )
                    session.commit()
                    print("Existing risk factors deleted.")
                else:
                    print("Skipping insertion. Existing risk factors remain unchanged.")
                    return

        rows_to_insert: List[RiskFactor] = []
        for order_index, factor in enumerate(factors, start=1):
            factor_text = (factor.get("text") or "").strip()
            if not factor_text:
                continue

            rows_to_insert.append(
                RiskFactor(
                    document_id=document_id,
                    risk_factor=factor.get("risk_factor"),
                    text=factor_text,
                    idx=order_index,
                )
            )

        if not rows_to_insert:
            print(f"No valid factors to insert for '{document_id}'.")
            return

        session.add_all(rows_to_insert)
        session.commit()
        print(
            f"✓ Saved {len(rows_to_insert)} risk factors for document '{document_id}' successfully."
        )

    except Exception as exc:
        session.rollback()
        print(f"Error saving risk factors: {exc}")
        raise
    finally:
        session.close()

def main(config: GlobalConfig, on_existing: str = "prompt") -> None:
    """
    Save one configured document to the database with duplicate handling control.

    Args:
        config: Global configuration object.
        on_existing: Duplicate handling policy for existing document IDs.
            One of "prompt", "skip", "delete".

    Returns:
        None.
    """
    init_db(config.POSTGRESQL_URL, expected_tables={"documents", "risk_factors"})
    print("Database is ready for use.")
    database_url = config.POSTGRESQL_URL
    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    chunks_file = load_chunks(entry.chunks_dir / f"{entry.doc_id}.json")
    engine = create_engine(database_url, echo=False)
    
    # Save document to database
    save_factors_to_db(
        chunks_file.chunks,
        entry.doc_id,
        engine,
        on_existing=on_existing,
    )
    
    engine.dispose()

