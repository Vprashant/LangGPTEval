from .factories import EvaluationFactory
from .models import EvaluationInput, EvaluationResult, ModelWrapper
from typing import Any

def evaluate_faithfulness(input_data: EvaluationInput, model: Any) -> EvaluationResult:
    try:
        evaluator = EvaluationFactory.create_faithfulness_evaluator(model)
        return evaluator(input_data.context, input_data.response)
    except Exception as e:
        raise ValueError(f"Failed to evaluate faithfulness: {str(e)}")

def evaluate_context_recall(input_data: EvaluationInput, model: Any) -> EvaluationResult:
    try:
        evaluator = EvaluationFactory.create_context_recall_evaluator(model)
        return evaluator(input_data.context, input_data.response)
    except Exception as e:
        raise ValueError(f"Failed to evaluate context recall: {str(e)}")

def evaluate_answer_relevancy(input_data: EvaluationInput, model: Any) -> EvaluationResult:
    try:
        evaluator = EvaluationFactory.create_answer_relevancy_evaluator(model)
        return evaluator(input_data.context, input_data.response)
    except Exception as e:
        raise ValueError(f"Failed to evaluate answer relevancy: {str(e)}")

def evaluate_context_relevancy(input_data: EvaluationInput, model: Any) -> EvaluationResult:
    try:
        evaluator = EvaluationFactory.create_context_relevancy_evaluator(model)
        return evaluator(input_data.context, input_data.response)
    except Exception as e:
        raise ValueError(f"Failed to evaluate context relevancy: {str(e)}")
