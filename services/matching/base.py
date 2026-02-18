from abc import ABC, abstractmethod
from typing import List, Tuple
from schemas.product_schema import Product


# Abstract base class for matchers - defines the interface for ranking products based on a query
class BaseMatcher(ABC):

    @abstractmethod
    def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
        """
        Returns products sorted by relevance score
        """
        pass
