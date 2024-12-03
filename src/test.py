import os
from dotenv import load_dotenv
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import Pipeline
import google.generativeai as genai

load_dotenv()

class GenAICog:
    def __init__(self):
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

    def process_message(self, message):
        # Add message to chat history
        self.add_to_chat_history(message, "User")

        # Use Haystack to search for relevant content
        search_results = self.pipeline.run(query=message, params={"Retriever": {"top_k": 5}})
        search_texts = [doc.content for doc in search_results["documents"]]

        # Combine search results and user message for AI response
        combined_input = f"User: {message}\n\nRelevant Information:\n" + "\n".join(search_texts)

        # Generate AI response
        ai_response = self.model.generate_content(combined_input).text

        # Optionally, add AI response to chat history
        self.add_to_chat_history(ai_response, "AI")

        return ai_response

if __name__ == "__main__":
    cog = GenAICog()
    print("GenAICog initialized. Type your messages below (type 'exit' to quit):")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = cog.process_message(user_input)
        print(f"AI: {response}")