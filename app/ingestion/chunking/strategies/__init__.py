from typing import Annotated, Union
from pydantic import Field

from .ChunkStrategyFixedSize import ChunkStrategyFixedSize
from .ChunkStrategySentence import ChunkStrategySentence
from .ChunkStrategySentenceMetadata import ChunkStrategySentenceMetadata 
from .ChunkStrategyParagraph import ChunkStrategyParagraph
from .ChunkStrategyParagraphMetadata import ChunkStrategyParagraphMetadata 

ChunkStrategy = Annotated[
    Union[ChunkStrategyFixedSize, ChunkStrategySentence, ChunkStrategySentenceMetadata, ChunkStrategyParagraph, ChunkStrategyParagraphMetadata],   
    Field(discriminator="method"),
]