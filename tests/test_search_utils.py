import pytest
import pandas as pd
from src.search_utils import clean_series

def test_clean_series():
    """
    Tests the clean_series function from search_utils.
    """
    # Define some example data
    raw_data = [
        "Test <p>HTML</p>", "Extra   spaces", "New\nlines\n\n", None, "  Trim me  "        
    ]
    series = pd.Series(raw_data, dtype=str)
    
    # Define the expected output
    expected_data = [
        "Test HTML", "Extra spaces", "New lines", "", "Trim me"
    ]
    
    # Get the real output
    cleaned = clean_series(series)
    
    # Test
    pd.testing.assert_series_equal(cleaned, pd.Series(expected_data, dtype=str))

def test_preprocess_batch():
    """
    Tests the preprocess_batch function.
    """
    from src.search_utils import preprocess_batch
    
    data = {
        'id': [1],
        'short_description': ['My Title'],
        'content': ['Some details.'],
        'category': ['Software'],
        'subcategory': ['Email'],
        'software/system': ['Outlook']
    }
    df = pd.DataFrame(data)
    
    result_series = preprocess_batch(df)
    
    expected_string = "Title: My Title | Category: Software Email | Software: Outlook | Content: Some details."
    
    assert len(result_series) == 1
    assert result_series.iloc[0] == expected_string