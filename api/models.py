from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, List

class SearchRequest(BaseModel):
    query: str
    n_items: Optional[int] = 10

# How the response for a ticket should be sent
class TicketRequest(BaseModel):
    short_description: str
    category: Optional[str] = "N/D"
    subcategory: Optional[str] = "N/D"
    content: Optional[str] = "N/D"
    software: Optional[str] = "N/D"

# How a ticket is returned from the API
class TicketResponse(BaseModel):
    id: int
    short_description: str
    category: Optional[str]
    subcategory: Optional[str]
    content: Optional[str]
    software: Optional[str]

    model_config = ConfigDict(from_attributes=True)

# Ticket + Similarity Score
class SearchHit(BaseModel):
    ticket: TicketResponse
    score: float  # similarity score

# Return search results
class SearchResult(BaseModel):
    items: List[SearchHit]
    count: int
    search_time: float  # DB search only
    total_time: float   # total endpoint time