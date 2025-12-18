
import re


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
