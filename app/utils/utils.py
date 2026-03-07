import re
from sqlalchemy import create_engine, inspect


def slugify_name(company: str) -> str:
    """
    Qdrant collection names should be simple. This keeps letters, digits, _ and -.
    """
    s = company.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    if not s:
        raise ValueError("Company name became empty after sanitization.")
    return f"company__{s}"



def init_db(database_url: str, expected_tables: set) -> None:
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
