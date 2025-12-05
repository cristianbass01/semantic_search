"""
Preprocessing module

Defines the functions used to preprocess data before embedding.
"""

import pandas as pd

def clean_series(series: pd.Series) -> pd.Series:
    """Cleans a pandas Series of text data.

    This function is optimized for semantic models:
    - Removes HTML tags.
    - Normalizes all whitespace (newlines, tabs, etc.) to a single space.
    - Trims leading/trailing whitespace.
    - Keeps punctuation and casing, as they are important for semantic meaning.
    
    Args:
        series: The input pandas Series containing text.

    Returns:
        A new pandas Series with the cleaned text.
    """
    s_cleaned = series.fillna('').astype(str)
    s_cleaned = s_cleaned.str.replace(r'<[^>]+>', ' ', regex=True)  # Remove HTML
    s_cleaned = s_cleaned.str.replace(r'\s+', ' ', regex=True)     # Normalize whitespace
    s_cleaned = s_cleaned.str.strip()                             # Trim
    return s_cleaned

def preprocess_batch(df_batch: pd.DataFrame) -> pd.Series:
    """Converts a DataFrame of ticket data into a single text Series for embedding.

    This function formats multiple columns into a single descriptive string
    for each ticket, which is then fed to the embedding model.

    Args:
        df_batch: A DataFrame containing ticket data. Must include 'id'
            and is expected to have 'short_description', 'content', 'category',
            'subcategory', and 'software/system'.

    Returns:
        A pandas Series where each item is the formatted string
        ready for embedding.
    """
    
    batch_index = df_batch.index
    batch_columns = df_batch.columns

    def get_col_safe(col_name: str) -> pd.Series:
        """Helper to safely get a column or return an empty Series."""
        if col_name in batch_columns:
            return df_batch[col_name]
        else:
            # Return an empty Series with the same index
            return pd.Series(dtype=str).reindex(batch_index)

    # Clean each data column individually
    s_cat = clean_series(get_col_safe('category'))
    s_sub = clean_series(get_col_safe('subcategory'))
    s_desc = clean_series(get_col_safe('short_description'))
    s_cont = clean_series(get_col_safe('content'))
    s_sw = clean_series(get_col_safe('software/system'))

    # Combine into the final formatted string
    # This format is designed to give the model context
    final_text_series = (
        "Title: " + s_desc + " | " +
        "Category: " + s_cat + " " +  s_sub + " | " +
        "Software: " + s_sw + " | " +
        "Content: " + s_cont
    )
    
    return final_text_series