import os
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from dotenv import load_dotenv
from haystack.utils import Secret

load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_key = Secret.from_env_var("AZURE_OPENAI_API_KEY")
model = 'gpt-35-turbo-16k'

client = AzureOpenAIChatGenerator(api_key=api_key, azure_endpoint=azure_endpoint, azure_deployment=model)
response = client.run(
	  [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
)
print(response)