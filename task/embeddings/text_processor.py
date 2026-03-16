from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


# Determines how vector similarity is measured in the DB query
class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Straight-line distance in vector space (<->); sensitive to magnitude
    COSINE_DISTANCE = "cosine"        # Angle between vectors (<=>); magnitude-independent, better for text


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Open a new psycopg2 connection to the pgvector PostgreSQL database"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
        self,
        file_name: str,
        chunk_size: int,
        overlap: int,
        dimensions: int,
        truncate: bool = False,
    ) -> None:
        # Read the entire document into memory
        with open(file_name, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the document into overlapping fixed-size character chunks
        chunks = chunk_text(content, chunk_size, overlap)
        # Send all chunks to the embeddings API in one batch; returns {index: vector}
        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                if truncate:
                    # Remove all existing rows so re-indexing starts clean
                    cur.execute("TRUNCATE TABLE vectors")

                for i, chunk in enumerate(chunks):
                    # pgvector expects the vector as a string like "[0.1, 0.2, ...]"
                    embedding_str = str(embeddings[i])
                    cur.execute(
                        "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                        (file_name, chunk, embedding_str),
                    )
            conn.commit()
        finally:
            conn.close()

    def search(
        self,
        mode: SearchMode,
        user_request: str,
        top_k: int,
        min_score: float,
        dimensions: int,
    ) -> list[str]:
        # Embed the user's question so it can be compared against stored chunk vectors
        embeddings = self.embeddings_client.get_embeddings([user_request], dimensions)
        embedding_str = str(embeddings[0])

        # Pick the pgvector distance operator based on the chosen search mode
        operator = "<->" if mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"

        # Retrieve chunks whose distance to the query vector is within min_score,
        # ordered closest-first, limited to top_k results
        query = f"""
            SELECT text, (embedding {operator} %s::vector) AS distance
            FROM vectors
            WHERE (embedding {operator} %s::vector) <= %s
            ORDER BY distance
            LIMIT %s
        """

        conn = self._get_connection()
        try:
            # RealDictCursor returns rows as dicts so we can access columns by name
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (embedding_str, embedding_str, min_score, top_k))
                results = cur.fetchall()
            # Return only the raw text of each matching chunk, not the distance scores
            return [row["text"] for row in results]
        finally:
            conn.close()

