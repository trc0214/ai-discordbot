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
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator, GoogleAIGeminiGenerator
from haystack.components.converters import OutputAdapter
from itertools import chain
from typing import Any, List

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai_model = 'gemini-1.5-flash'
os.environ["GOOGLE_API_KEY"] = api_key
ai_chat_channel_id = 1314151629796151307


class ListJoiner:
    def __init__(self, _type: Any):
        self._type = _type

    def run(self, values: List[Any]):
        result = list(chain(*values))
        return {"values": result}

def initialize_pipeline():
    # Initialize document and memory stores
    document_store = InMemoryDocumentStore()
    memory_store = InMemoryChatMessageStore()
    memory_retriever = ChatMessageRetriever(memory_store)
    memory_writer = ChatMessageWriter(memory_store)

    # Define query rephrase template
    query_rephrase_template = """
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
    """
    
    # Initialize pipeline and add components
    pipeline = Pipeline()
    pipeline.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
    pipeline.add_component("query_rephrase_llm", GoogleAIGeminiGenerator())
    pipeline.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store, top_k=3))
    pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
    pipeline.add_component("llm", GoogleAIGeminiChatGenerator())
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

class ConversationalGenaiCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.allowed_channels = [ai_chat_channel_id]
        self.pipeline = initialize_pipeline()
        self.system_message_template = "You are a helpful AI assistant using provided supporting documents and conversation history to assist humans"
        self.user_message_template = """Given the conversation history and the provided supporting documents, give a brief answer to the question.
            Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.

                Conversation history:
                {% for memory in memories %}
                    {{ memory.content }}
                {% endfor %}

                Supporting documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                \nQuestion: {{query}}
                \nAnswer:
            """

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user or message.channel.id not in self.allowed_channels:
            return

        question = message.content
        system_message = ChatMessage.from_system(self.system_message_template)
        user_message = ChatMessage.from_user(self.user_message_template)
        messages = [system_message, user_message]
        res = self.pipeline.run(data={"query_rephrase_prompt_builder": {"query": question},
                                      "prompt_builder": {"template": messages, "query": question},
                                      "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                                include_outputs_from=["llm", "query_rephrase_llm"])
        assistant_resp = res['llm']['replies'][0].content
        await message.reply(assistant_resp)

async def setup(bot):
    await bot.add_cog(ConversationalGenaiCog(bot))

if __name__ == "__main__":
    # just for testing
    pipeline = initialize_pipeline()
    system_message_template = "You are a helpful AI assistant using provided supporting documents and conversation history to assist humans"
    user_message_template = """Given the conversation history and the provided supporting documents, give a brief answer to the question.
        Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.

            Conversation history:
            {% for memory in memories %}
                {{ memory.content }}
            {% endfor %}

            Supporting documents:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
        """
        
    def on_message(message):
        question = message.content
        system_message = ChatMessage.from_system(system_message_template)
        user_message = ChatMessage.from_user(user_message_template)
        messages = [system_message, user_message]
        res = self.pipeline.run(data={"query_rephrase_prompt_builder": {"query": question},
                                      "prompt_builder": {"template": messages, "query": question},
                                      "memory_joiner": {"values": [ChatMessage.from_user(question)]}},
                                include_outputs_from=["llm", "query_rephrase_llm"])
        assistant_resp = res['llm']['replies'][0].content
        print(assistant_resp)

    while True:
        message = input("Enter a message: ")
        on_message(message)
