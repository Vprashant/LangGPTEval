from pydantic import BaseModel, validator
from typing import List, Any

class ContextData(BaseModel):
    page_content: str

class EvaluationInput(BaseModel):
    context: List[ContextData]
    response: str

    @validator('context', each_item=True)
    def check_page_content(cls, value):
        if not value.page_content:
            raise ValueError("Context page content cannot be empty")
        return value

class EvaluationResult(BaseModel):
    score: str

class ModelWrapper(BaseModel):
    model: Any

    def invoke(self, prompt: str) -> str:
        if hasattr(self.model, 'invoke'):
            return self.model.invoke(prompt)
        else:
            raise ValueError("The provided model does not have an 'invoke' method")
