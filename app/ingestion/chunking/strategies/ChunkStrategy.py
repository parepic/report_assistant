from pydantic import BaseModel
from typing import List


class ChunkStrategy(BaseModel):
    method: str

    def create_chunks(self, text: str) -> List[str]:
        """
        Placeholder for chunk creation logic.
        """
        pass
