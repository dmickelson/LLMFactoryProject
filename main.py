# Example of how to use the LLM Factory

# from llmfactory.llm_factory import CompletionModel, LLMFactory
from llmfactory import llm_factory

if __name__ == "__main__":

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "If it takes 2 hours to dry 1 shirt out in the sun, how long will it take to dry 5 shirts?",
        },
    ]

    llm = llm_factory.LLMFactory("cohere")
    completion = llm.create_completion(
        response_model=llm_factory.CompletionModel,
        messages=messages,
    )
    assert isinstance(completion, llm_factory.CompletionModel)

    print(f"Response: {completion.response}\n")
    print(f"Reasoning: {completion.reasoning}")
