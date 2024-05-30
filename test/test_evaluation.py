from LangGPTEval.models import EvaluationInput, ContextData
from LangGPTEval.evaluation import evaluate_faithfulness, evaluate_context_recall, evaluate_answer_relevancy, evaluate_context_relevancy
from langchain_openai import OpenAI


# Replace 'your-openai-api-key' with your actual OpenAI API key
api_key = 'your-openai-api-key'


openai_model =  OpenAI(openai_api_key=api_key, openai_organization="YOUR_ORGANIZATION_ID")
# Example and test with your mode api
context = [ContextData(page_content="Test context")]
response = "Test response"
input_data = EvaluationInput(context=context, response=response)

try:
    # Evaluate with the LangChain OpenAI model
    faithfulness_result = evaluate_faithfulness(input_data, openai_model)
    context_recall_result = evaluate_context_recall(input_data, openai_model)
    answer_relevancy_result = evaluate_answer_relevancy(input_data, openai_model)
    context_relevancy_result = evaluate_context_relevancy(input_data, openai_model)

    print(faithfulness_result.score)
    print(context_recall_result.score)
    print(answer_relevancy_result.score)
    print(context_relevancy_result.score)
except ValueError as e:
    print(f"An error occurred during evaluation: {str(e)}")
