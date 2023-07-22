from typing import Any, List

from langchain.docstore.document import Document
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "ggrn/e5-small-v2"
local_embeddings = HuggingFaceEmbeddings(model_name=model_name)

class CustomSupabaseVectorStore(SupabaseVectorStore):
    """A custom vector store that uses the match_vectors table instead of the vectors table."""
    brain_id: str = "none"

    def __init__(
        self,
        client: Client,
        embedding: HuggingFaceEmbeddings,
        table_name: str,
        brain_id: str = "none",
    ):
        super().__init__(client, embedding, table_name)
        self.brain_id = brain_id

    def similarity_search(
        self,
        query: str,
        table: str = "match_vectors",
        k: int = 6,
        threshold: float = 0.5,
        **kwargs: Any
    ) -> List[Document]:
        print("Query: ", (query))
        # vectors = self._embedding.embed_documents([query])
        vectors = local_embeddings.embed_query(query)
        print("LENGHT: ", len(vectors))  #  768?
        query_embedding = vectors #vectors[0]
        res = self._client.rpc(
            table,
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "p_brain_id": str(self.brain_id),
            },
        ).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        documents = [doc for doc, _ in match_result]

        return documents
