from typing import Annotated, Union
from pydantic import Field

from .ChunkStrategyFixedSize import ChunkStrategyFixedSize
from .ChunkStrategySentence import ChunkStrategySentence

ChunkStrategy = Annotated[
    Union[ChunkStrategyFixedSize, ChunkStrategySentence],   
    Field(discriminator="method"),
]