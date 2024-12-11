import os
from dotenv import load_dotenv
from discord.ext import commands
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators import AzureOpenAIGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.components.converters import OutputAdapter
from itertools import chain
from typing import Any, List
from haystack import component
from haystack.core.component.types import Variadic
from haystack.utils import Secret

load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
chat_api_key = Secret.from_env_var("AZURE_OPENAI_API_KEY")
model = 'gpt-35-turbo-16k'
ai_chat_channel_id = 1315587354538545202

# Initialize the generators with the correct parameters
llm = AzureOpenAIGenerator(azure_endpoint=azure_endpoint, api_key=api_key, azure_deployment=model)
chat_llm = AzureOpenAIChatGenerator(azure_endpoint=azure_endpoint, api_key=chat_api_key, azure_deployment=model)

# template
query_rephrase_template = """
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
    """
system_message_template = "You are a friendly and helpful AI assistant. Engage in a casual and informative conversation with the user."
user_message_template = """Given the conversation history and the provided supporting documents, respond to the user's question in a friendly manner.

            Conversation history:
            {% for memory in memories %}
                {{ memory.content }}
            {% endfor %}

            Supporting documents:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nUser name: {{ user_name }}
            \nUser message: {{query}}
            \nCurrent time: {{ current_time }}
            \nResponse:
        """

@component
class ListJoiner:
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}

def initialize_pipeline():
    # Initialize document and memory stores
    document_store = InMemoryDocumentStore()
    memory_store = InMemoryChatMessageStore()
    memory_retriever = ChatMessageRetriever(memory_store)
    memory_writer = ChatMessageWriter(memory_store)

    # Define query rephrase template
    
    # Initialize pipeline and add components
    pipeline = Pipeline()
    pipeline.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
    pipeline.add_component("query_rephrase_llm", llm)
    pipeline.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories","user_name","current_time"], required_variables=["query", "documents", "memories", "user_name","current_time"]))
    pipeline.add_component("llm", chat_llm)
    pipeline.add_component("memory_retriever", memory_retriever)
    pipeline.add_component("memory_writer", memory_writer)
    pipeline.add_component("memory_joiner", ListJoiner(List[ChatMessage]))

    # Connect components
    pipeline.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
    pipeline.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
    pipeline.connect("query_rephrase_llm.replies", "list_to_str_adapter")
    pipeline.connect("list_to_str_adapter", "retriever.query")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "memory_joiner")
    pipeline.connect("memory_joiner", "memory_writer")
    pipeline.connect("memory_retriever", "prompt_builder.memories")

    return pipeline

class AzureOpenAIConversationalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.allowed_channels = [ai_chat_channel_id]
        self.pipeline = initialize_pipeline()
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user or message.channel.id not in self.allowed_channels:
            return

        question = message.content
        user_name = message.author.display_name
        system_message = ChatMessage.from_system(self.system_message_template)
        user_message = ChatMessage.from_user(self.user_message_template)
        messages = [system_message, user_message]
        res = self.pipeline.run(data={"query_rephrase_prompt_builder": {"query": question},
                                      "prompt_builder": {"template": messages,"user_name": user_name, "query": question, "current_time": message.created_at},
                                      "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                                include_outputs_from=["llm", "query_rephrase_llm"])
        assistant_resp = res['llm']['replies'][0].content
        await message.reply(assistant_resp)

async def setup(bot):
    await bot.add_cog(AzureOpenAIConversationalCog(bot))


# just for testing
if __name__ == "__main__":
    pipeline = initialize_pipeline()
        
    def on_message(message):
        question = message
        system_message = ChatMessage.from_system(system_message_template)
        user_message = ChatMessage.from_user(user_message_template)
        messages = [system_message, user_message]
        res = pipeline.run(data={"query_rephrase_prompt_builder": {"query": question},
                                      "prompt_builder": {"template": messages,"user_name": "tim", "query": question, "current_time": "2022-02-02 12:00:00"},
                                      "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                                include_outputs_from=["llm", "query_rephrase_llm"])
        assistant_resp = res.get('llm', {}).get('replies', [ChatMessage.from_system("No response")])[0].content
        print(assistant_resp)

    on_message("where is the capital of france?")