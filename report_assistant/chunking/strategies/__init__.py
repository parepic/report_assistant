from typing import Annotated, Union
from pydantic import Field

from .ChunkStrategyFixedSize import ChunkStrategyFixedSize
from .ChunkStrategySentence import ChunkStrategySentence
from .ChunkStrategySentenceMetadata import ChunkStrategySentenceMetadata 

ChunkStrategy = Annotated[
    Union[ChunkStrategyFixedSize, ChunkStrategySentence, ChunkStrategySentenceMetadata],   
    Field(discriminator="method"),
]