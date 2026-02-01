import re
from typing import Tuple


def get_heading_level_pattern(text: str) -> Tuple[re.Pattern, re.Pattern]:
    """
    Determine section and subsection heading patterns based on the first heading level found.
    
    If first heading is single '#' (level 1):
      - Section is '#'
      - Subsection is '##'
    
    If first heading is '##' (level 2):
      - Section is '##'
      - Subsection is '###'
    
    Returns tuple of (section_pattern, subsection_pattern)
    """
    # Find first line with heading markers
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # Count leading hashes
            hash_count = len(stripped) - len(stripped.lstrip('#'))
            
            if hash_count == 1:
                # First heading is single #, so section is #, subsection is ##
                section_pattern = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
                subsection_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
            else:
                # First heading is ##, so section is ##, subsection is ###
                section_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
                subsection_pattern = re.compile(r"^\s*###\s+(.+?)\s*$", re.MULTILINE)
            
            return section_pattern, subsection_pattern
    
    # Default to # for section and ## for subsection if no headings found
    section_pattern = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
    subsection_pattern = re.compile(r"^\s*##\s+(.+?)\s*$", re.MULTILINE)
    return section_pattern, subsection_pattern
