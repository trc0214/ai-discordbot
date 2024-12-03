import os
import asyncio
from discord.ext import commands
import google.generativeai as genai
from dotenv import load_dotenv
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import Pipeline

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
        self.document_store = InMemoryDocumentStore(use_bm25=True)
        self.retriever = BM25Retriever(document_store=self.document_store)
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])

        # Chat history
        self.chat_history = []

    def add_to_chat_history(self, message, author):
        self.chat_history.append({"content": message, "author": author})
        self.document_store.write_documents([{"content": message, "meta": {"author": author}}])

    def process_message(self, message, author):
        # Add message to chat history
        self.add_to_chat_history(message, author)

        # Use Haystack to search for relevant content
        search_results = self.pipeline.run(query=message, params={"Retriever": {"top_k": 3}})
        search_texts = [doc.content for doc in search_results["documents"]]

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