import os
import asyncio
from discord.ext import commands
import google.generativeai as genai
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever

load_dotenv()

class GenAICog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.allowed_channels = [1303271554897154069]  # Replace with your actual allowed channel IDs
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API_KEY found. Please set the GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # Initialize Haystack components
        self.document_store = MongoDBAtlasDocumentStore(
            database_name="haystack_test",
            collection_name="test_collection",
            vector_search_index="embedding_index",
        )
        self.doc_writer = DocumentWriter(document_store=self.document_store, policy=DuplicatePolicy.SKIP)
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
        self.query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2")

        # Pipeline that ingests document for retrieval
        self.indexing_pipe = Pipeline()
        self.indexing_pipe.add_component(instance=self.doc_embedder, name="doc_embedder")
        self.indexing_pipe.add_component(instance=self.doc_writer, name="doc_writer")
        self.indexing_pipe.connect("doc_embedder.documents", "doc_writer.documents")

        # RAG pipeline
        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component(instance=self.query_embedder, name="query_embedder")
        self.rag_pipeline.add_component(instance=MongoDBAtlasEmbeddingRetriever(document_store=self.document_store), name="retriever")
        self.rag_pipeline.add_component(instance=PromptBuilder(template=self.prompt_template()), name="prompt_builder")
        self.rag_pipeline.connect("query_embedder", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")

        # Chat history
        self.chat_history = []

    def prompt_template(self):
        return """
        Given these documents, answer the question.\nDocuments:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        \nQuestion: {{question}}
        \nAnswer:
        """

    def add_to_chat_history(self, message, author):
        self.chat_history.append({"content": message, "author": author})
        documents = [Document(content=message, meta={"author": author})]
        self.indexing_pipe.run({"doc_embedder": {"documents": documents}})

    def process_message(self, message, author):
        # Add message to chat history
        self.add_to_chat_history(message, author)

        # Use Haystack to search for relevant content
        question = message
        result = self.rag_pipeline.run(
            {
                "query_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )
        search_texts = [doc.content for doc in result["documents"]]

        # Combine search results and user message for AI response
        combined_input = f"{author}: {message}\n\nRelevant Information:\n" + "\n".join(search_texts) + "\n\nAI, please provide a concise response without repeating the whole 'Relevant Information' section."

        # Generate AI response
        ai_response = self.model.generate_content(combined_input).text

        # Optionally, add AI response to chat history
        self.add_to_chat_history(ai_response, "AI")

        return ai_response

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.channel.id not in self.allowed_channels:
            return

        try:
            response = await asyncio.to_thread(self.process_message, message.content, message.author.name)
            await message.channel.send(response)
        except Exception as e:
            print(f"Error generating response: {e}")
            await message.channel.send("Sorry, something went wrong. Please try again later.")

async def setup(bot):
    await bot.add_cog(GenAICog(bot))

if __name__ == "__main__":
    # test without bot
    cog = GenAICog(None)
    message = "Where does Mark live"
    response = cog.process_message(message, "User")