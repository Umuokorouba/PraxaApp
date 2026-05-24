from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

os.environ["OPENROUTER_API_KEY"] = "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"

class ChatModel(ChatOpenAI):
    """
    Creates a chat model from openrouter.ai using the OpenAI API
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str] = None,
            openai_api_base: str="https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = "google/gemma-3-27b-it:free") -> ChatModel:
    """
    Gets a reference to a model
    
    :param model_name: Name of the model
    :type model_name: str
    :return: the model
    :rtype: ChatModel
    """
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0
    )

if __name__ == "__main__":
# when run as a script, run some tests to demonstrate capabilities
#    model = get_model()
#    from langchain_core.messages import HumanMessage
#    from langchain.prompts import ChatPromptTemplate

#    ???
#    ???
#    ???
#    ???

#    response = model.invoke(
#        [???("You are a helpful assistant."),
#         ???("What are some plays by Tawfiq al-Hakim?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [???("You are a helpful assistant."),
#         ???("What is Ryan Calais Camerons's most recent play?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [???("You are a helpful assistant."),
#         ???("What Broadway shows have more than 10,000 performances?")])
#    print(response.content)

#    print(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    response = model.invoke(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    print(response.content)

#    chain = ???
#    response = ???{"playwright": "Ryan Calais Cameron"})
#    print(response.content)

    pass
