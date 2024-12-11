import os
from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import AzureOpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from dotenv import load_dotenv
from haystack.utils import Secret

load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
#api_key = Secret.from_env_var("AZURE_OPENAI_API_KEY")
model = 'gpt-35-turbo-16k'
llm = AzureOpenAIGenerator(azure_endpoint=azure_endpoint, api_key=api_key, azure_deployment=model)


docstore = InMemoryDocumentStore()
docstore.write_documents([Document(content="Rome is the capital of Italy"), Document(content="Paris is the capital of France")])

query = "What is the capital of France?"

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""
pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", llm)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

res=pipe.run({
    "prompt_builder": {
        "query": query
    },
    "retriever": {
        "query": query
    }
})

print(res)