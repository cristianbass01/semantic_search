from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """
    Cleans a single string of text for semantic models.
    
    - Removes HTML tags.
    - Normalizes all whitespace (newlines, tabs, etc.) to a single space.
    - Trims leading/trailing whitespace.
    - Keeps punctuation and casing.
    
    Args:
        text: The input text string.
    
    Returns:
        The cleaned text string.
    """
    if text is None:
        text = ''
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)   # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)       # Normalize whitespace
    text = text.strip()                     # Trim
    return text

def preprocess_ticket(ticket: Dict) -> str:
    """
    Converts a single ticket dict into a single formatted string for embedding.
    
    Expected keys in ticket dict:
        - short_description
        - content
        - category
        - subcategory
        - software
    
    Missing keys are treated as empty strings.
    
    Returns:
        A formatted, cleaned string ready for embedding.
    """
    # Safely get each field, defaulting to empty string
    short_description = clean_text(ticket.get("short_description", ""))
    content = clean_text(ticket.get("content", ""))
    category = clean_text(ticket.get("category", ""))
    subcategory = clean_text(ticket.get("subcategory", ""))
    software = clean_text(ticket.get("software", ""))

    # Combine into a single descriptive string
    formatted_text = (
        f"Title: {short_description} | "
        f"Category: {category} {subcategory} | "
        f"Software: {software} | "
        f"Content: {content}"
    )

    return formatted_text
