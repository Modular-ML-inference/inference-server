from typing import Optional

from pydantic import BaseModel


class TrainingConfiguration(BaseModel):
    model: str
    epochs: int
    rounds: int
    optimizer: str
    strategy: str
    server_address: Optional[str] = "127.0.0.1:8080"
    batch_size: Optional[int] = 32
    steps_per_epoch: Optional[int] = 3

    class Config:
        arbitrary_types_allowed = True