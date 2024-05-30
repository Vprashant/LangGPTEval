from langchain.prompts import PromptTemplate

def get_faithfulness_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the faithfulness of the response to the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explanation.\n"
            "Score:"
        )
    )

def get_context_recall_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate how well the response recalls the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explanation.\n"
            "Score:"
        )
    )

def get_answer_relevancy_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the relevancy of the response to the question on a scale from 0 to 1. be very precise give me only the score as output. do return any explanation.\n"
            "Score:"
        )
    )

def get_context_relevancy_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "response"],
        template=(
            "Context: {context}\n"
            "Response: {response}\n"
            "Evaluate the relevancy of the response to the context on a scale from 0 to 1. be very precise give me only the score as output. do return any explanation.\n"
            "Score:"
        )
    )
