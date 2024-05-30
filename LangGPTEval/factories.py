from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from .models import EvaluationInput, EvaluationResult, ModelWrapper
from .prompts import get_faithfulness_prompt, get_context_recall_prompt, get_answer_relevancy_prompt, get_context_relevancy_prompt

class EvaluationFactory:
    @staticmethod
    def create_faithfulness_evaluator(model):
        return EvaluationFactory._create_evaluator(get_faithfulness_prompt, model)

    @staticmethod
    def create_context_recall_evaluator(model):
        return EvaluationFactory._create_evaluator(get_context_recall_prompt, model)

    @staticmethod
    def create_answer_relevancy_evaluator(model):
        return EvaluationFactory._create_evaluator(get_answer_relevancy_prompt, model)

    @staticmethod
    def create_context_relevancy_evaluator(model):
        return EvaluationFactory._create_evaluator(get_context_relevancy_prompt, model)

    @staticmethod
    def _create_evaluator(prompt_getter, model):
        model_wrapper = ModelWrapper(model=model)
        def evaluator(context, response):
            try:
                context_str = "\n---\n".join([d.page_content for d in context])
                prompt = prompt_getter()
                retrieval = RunnableParallel({"context": lambda x: context_str, "response": RunnablePassthrough()})
                chain = retrieval | prompt | model_wrapper | StrOutputParser()
                score = chain.invoke(response)
                return EvaluationResult(score=score)
            except Exception as e:
                raise ValueError(f"An error occurred during evaluation: {str(e)}")

        return evaluator
