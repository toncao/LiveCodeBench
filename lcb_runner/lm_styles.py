from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"

    GenericBase = "GenericBase"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_repr": self.model_repr,
            "model_style": self.model_style.value,
            "release_date": int(self.release_date.timestamp() * 1000),
            "link": self.link,
        }
        
LanguageModelList = []

LanguageModelStore: dict[str, LanguageModel] = {}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))
