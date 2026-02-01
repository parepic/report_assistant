from typing import Annotated, Union
from pydantic import Field

from .ChunkStrategyFixedSize import ChunkStrategyFixedSize
from .ChunkStrategySentence import ChunkStrategySentence
from .ChunkStrategySentenceMetadata import ChunkStrategySentenceMetadata 
from .ChunkStrategyParagraphMetadata import ChunkStrategyParagraphMetadata 

ChunkStrategy = Annotated[
    Union[ChunkStrategyFixedSize, ChunkStrategySentence, ChunkStrategySentenceMetadata, ChunkStrategyParagraphMetadata],   
    Field(discriminator="method"),
]