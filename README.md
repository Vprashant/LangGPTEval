# LangGPTEval
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPi version](https://img.shields.io/pypi/v/fancybbox)](https://pypi.org/project/fancybbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LangGPTEval**, Evaluation library designed for Retrieval-Augmented Generation (RAG) responses.Evaluates the faithfulness, context recall, answer relevancy, and context relevancy of responses generated by various models, including OpenAI, Azure, and custom models. With a complex architecture and advanced Pydantic validation, LangGPTEval ensures reliable and accurate evaluation metrics.

## 🌟 Introduction

LangGPTEval is designed to evaluate the quality of responses generated by RAG models. It supports multiple metrics for evaluation:

- **Faithfulness**: How true the response is to the given context.
- **Context Recall**: How well the response recalls the given context.
- **Answer Relevancy**: How relevant the response is to the question.
- **Context Relevancy**: How relevant the response is to the context.

LangGPTEval is highly customizable, allowing users to plug in their models and tailor the evaluation to their specific needs.

## 🛠️ Installation

You can install LangGPTEval using pip:

```bash
pip install LangGPTEval
```

## ⚡ Quick Start

Here’s a quick start guide to get you up and running with LangGPTEval.

1. **Install the library**.
2. **Prepare your data**.
3. **Evaluate your model**.

## 📚 Usage

### Importing the Library

First, import the necessary components from the LangGPTEval library.

```python
from LangGPTEval.models import EvaluationInput, ContextData
from LangGPTEval.evaluation import evaluate_faithfulness, evaluate_context_recall, evaluate_answer_relevancy, evaluate_context_relevancy
from langchain.llms import OpenAI
```

### Setting Up Your Model

Create an instance of your model. Here, we demonstrate using LangChain’s OpenAI model.

```python
class LangChainOpenAIModel:
    def __init__(self, api_key: str):
        self.llm = OpenAI(api_key=api_key)

    def invoke(self, prompt: Any) -> str:
        response = self.llm(prompt)
        score = response.strip()
        return score
```

### Example Data

Prepare the input data for evaluation.

```python
context = [ContextData(page_content="Test context")]
response = "Test response"
input_data = EvaluationInput(context=context, response=response)
```

### Evaluating the Model

Use the evaluation functions to evaluate the model’s performance.

```python
# Replace 'your-openai-api-key' with your actual OpenAI API key
api_key = 'your-openai-api-key'
openai_model = LangChainOpenAIModel(api_key)

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
```

## 🔍 Examples

### Example with Custom Model

```python
class CustomModel:
    def invoke(self, prompt):
        # Custom model implementation
        return "0.9"  # Example score

# Create a custom model instance
custom_model = CustomModel()

try:
    # Evaluate with the custom model
    faithfulness_result = evaluate_faithfulness(input_data, custom_model)
    context_recall_result = evaluate_context_recall(input_data, custom_model)
    answer_relevancy_result = evaluate_answer_relevancy(input_data, custom_model)
    context_relevancy_result = evaluate_context_relevancy(input_data, custom_model)

    print(faithfulness_result.score)
    print(context_recall_result.score)
    print(answer_relevancy_result.score)
    print(context_relevancy_result.score)
except ValueError as e:
    print(f"An error occurred during evaluation: {str(e)}")
```

## 🤝 Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before making a pull request.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## 📜 License

LangGPTEval is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


**Happy Evaluating!** 🎉

LangGPTEval is here to make your RAG model evaluations precise and easy. If you have any questions or need further assistance, feel free to reach out to me.

---


