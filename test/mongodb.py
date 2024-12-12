from haystack import Pipeline, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.writers import DocumentWriter
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.utils import Secret

# Create some example documents
documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
]

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = MongoDBAtlasDocumentStore(
    mongo_connection_string=Secret.from_env_var("MONGODB_CONNECTION_STRING"),
    database="test",
)

# Define some more components
doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2")

# Pipeline that ingests document for retrieval
ingestion_pipe = Pipeline()
ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
ingestion_pipe.run({"doc_embedder": {"documents": documents}})

# Build a RAG pipeline with a Retriever to get relevant documents to 
# the query and a OpenAIGenerator interacting with LLMs using a custom prompt.
prompt_template = """
Given these documents, answer the question.\nDocuments:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}

\nQuestion: {{question}}
\nAnswer:
"""
rag_pipeline = Pipeline()
rag_pipeline.add_component(instance=query_embedder, name="query_embedder")
rag_pipeline.add_component(instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
rag_pipeline.connect("query_embedder", "retriever.query_embedding")
rag_pipeline.connect("embedding_retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Ask a question on the data you just added.
question = "Where does Mark live?"
result = rag_pipeline.run(
    {
        "query_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)

# For details, like which documents were used to generate the answer, look into the GeneratedAnswer object
print(result["answer_builder"]["answers"])