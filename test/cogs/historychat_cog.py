import os
from discord.ext import commands
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from dotenv import load_dotenv

load_dotenv()
ai_chat_channel_id = 1315186189602394183
api_key = os.getenv("GEMINI_API_KEY")
model = 'gemini-1.5-flash'

os.environ["GOOGLE_API_KEY"] = api_key
llm = GoogleAIGeminiGenerator(model=model)
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content=""), 
])
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

class GenAIHistoryCog(commands.cog):
    def __init__(self, bot):
        self.bot = bot
        self.bot_name - bot.user.name
        self.allowed_channels = [ai_chat_channel_id]
        self.llm = llm

