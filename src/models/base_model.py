from abc import ABC, abstractmethod
from typing import Any



class BaseModel(ABC):
    @abstractmethod
    def entrainer(self, X_train: Any, y_train: Any) -> None:
        pass

    @abstractmethod
    def evaluer(self, X_test: Any, y_test: Any) -> float:
        pass

    @abstractmethod
    def predire(self, X: Any) -> Any:
        pass

