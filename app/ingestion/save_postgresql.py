from __future__ import annotations

from app.data_classes import GlobalConfig
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from pathlib import Path

from app.utils.load_utils import get_index_path, load_document_entry
from app.models import Document, Base


def init_db(database_url: str) -> None:
    """
    Initialize the database by checking for existing tables and creating missing ones.
    
    Checks if the Document model exists in the database.
    Creates any missing tables using SQLAlchemy's Base.metadata.create_all().
    """
    
    from app.models import Base
    # Create engine
    engine = create_engine(database_url, echo=False)
    
    # Get inspector to check existing tables
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    # Expected tables from models
    expected_tables = {"documents"}

    # Check what's missing
    missing_tables = expected_tables - existing_tables
    
    if missing_tables:
        print(f"Missing tables: {', '.join(sorted(missing_tables))}")
        print("Creating missing tables...")
        # Create all tables defined in Base metadata
        Base.metadata.create_all(engine)
        print("Database initialization complete.")
    else:
        print("All tables already exist. No action needed.")
    
    engine.dispose()


def save_document_to_db(entry, engine) -> None:
    """
    Save a document entry to the database.
    
    Checks if document already exists. If it does, prompts user to delete or skip.
    Otherwise inserts the document.
    
    Args:
        entry: DocumentEntry object loaded from config
        engine: SQLAlchemy engine
    
    Returns:
        None
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if document with this id already exists
        existing_doc = session.query(Document).filter(Document.id == entry.doc_id).first()
        
        if existing_doc:
            print(f"\n⚠️  Document with id '{entry.doc_id}' already exists in the database.")
            response = input("Delete all and re-insert? (yes/no): ").strip().lower()
            
            if response in ('yes', 'y'):
                print("Deleting existing document and cascading deletions...")
                session.delete(existing_doc)
                session.commit()
                print("Existing document deleted.")
            else:
                print("Skipping insertion. Document remains unchanged.")
                session.close()
                return
        
        # Read markdown file before creating Document
        md_file_path = Path(entry.text_dir) / f"{entry.doc_id}.md"
        if md_file_path.exists():
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
        else:
            raise FileNotFoundError(f"Markdown file not found at: {md_file_path}")
        
        # Insert the new document
        doc = Document(
            id=entry.doc_id,
            company=entry.company,
            fiscal_year=entry.fiscal_year,
            source_file_path=str(entry.source_file_path),
            questions_file_path=str(entry.questions_file_path) if entry.questions_file_path else None,
            text_dir=str(entry.text_dir) if entry.text_dir else None,
            chunks_dir=str(entry.chunks_dir) if entry.chunks_dir else None,
            text = md_text
        )
        
        session.add(doc)
        session.commit()
        print(f"✓ Document '{entry.doc_id}' saved to database successfully.")
        
    except Exception as e:
        session.rollback()
        print(f"Error saving document: {e}")
        raise
    finally:
        session.close()


def main(config: GlobalConfig) -> None:
    """
    Save document to database.
    
    Args:
        config: Global configuration object
    
    Returns:
        None
    """
    init_db(config.POSTGRESQL_URL)
    print("Database is ready for use.")
    database_url = config.POSTGRESQL_URL
    index_path = get_index_path(config)
    entry = load_document_entry(config.report_id, index_path, config)
    print(f"Loaded document entry for {config.report_id}:")
    
    # Create engine
    engine = create_engine(database_url, echo=False)
    
    # Save document to database
    save_document_to_db(entry, engine)
    
    engine.dispose()