from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack.utils import Secret
import os
from dotenv import load_dotenv

load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_key = Secret.from_env_var("AZURE_OPENAI_API_KEY")
model = 'gpt-35-turbo-16k'

# no parameter init, we don't use any runtime template variables
prompt_builder = ChatPromptBuilder()
llm = AzureOpenAIChatGenerator(api_key=api_key, azure_endpoint=azure_endpoint, azure_deployment=model)

pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")
location = "Berlin"
messages = [ChatMessage.from_system("Always respond in German even if some input data is in other languages."),
            ChatMessage.from_user("Tell me about {{location}}")]
res = pipe.run(data={"prompt_builder": {"template_variables":{"location": location}, "template": messages}})

print(res)
