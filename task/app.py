import os

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# Instructs the LLM how to behave: stay on-topic, rely only on the retrieved context
SYSTEM_PROMPT = """You are a RAG-powered assistant specialized in microwave oven usage and troubleshooting.

Each user message is structured as follows:
1. RAG Context: Relevant passages retrieved from the microwave manual.
2. User Question: The actual question from the user.

Instructions:
- Use the provided RAG Context as your primary source of information when answering.
- Only answer questions related to microwave usage, operation, or troubleshooting that are covered by the provided context or conversation history.
- If a question is unrelated to microwave usage, is not addressed by the context, or falls outside the conversation history, politely decline and explain that you can only assist with microwave-related topics.
- Do not speculate or provide information beyond what the context and history contain.
"""

# Template that wraps every user turn: injects retrieved manual passages above the actual question
USER_PROMPT = """RAG Context:
{context}

User Question:
{question}
"""

# PostgreSQL (pgvector) connection settings — stores document chunks and their embedding vectors
_DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres",
}
_EMBEDDING_MODEL = "text-embedding-3-small-1"  # Model used to convert text → numeric vectors
_CHAT_MODEL = "gpt-4o-mini"                    # LLM used to generate answers
_DIMENSIONS = 1536                             # Vector size produced by the embedding model
_MANUAL_PATH = os.path.join(os.path.dirname(__file__), "embeddings", "microwave_manual.txt")

# Client that calls the DIAL embeddings API to produce vectors from text
embeddings_client = DialEmbeddingsClient(_EMBEDDING_MODEL, API_KEY)
# Client that calls the DIAL chat API to generate responses
chat_client = DialChatCompletionClient(_CHAT_MODEL, API_KEY)
# Orchestrates chunking, storing embeddings in the DB, and similarity search
text_processor = TextProcessor(embeddings_client, _DB_CONFIG)


def run_chat():
    print("Indexing microwave manual...")
    # Read the manual, split it into 300-char chunks (with 40-char overlap),
    # embed each chunk, and store them in the DB (truncating any previous data)
    text_processor.process_text_file(
        file_name=_MANUAL_PATH,
        chunk_size=300,
        overlap=40,          # Overlap prevents losing context at chunk boundaries
        dimensions=_DIMENSIONS,
        truncate=True,       # Wipe existing rows so re-runs start fresh
    )
    print("Indexing complete. RAG-powered Microwave Assistant is ready.")
    print("Type your question and press Enter (or 'quit' to exit).\n")

    # Conversation keeps the full message history sent to the LLM on every turn
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Embed the question and find the top-5 most similar manual chunks via cosine distance
        context_chunks = text_processor.search(
            mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=5,          # Return at most 5 passages
            min_score=0.5,    # Cosine distance threshold — higher means less similar
            dimensions=_DIMENSIONS,
        )
        # Combine retrieved passages into a single context block (or signal none were found)
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

        # Build the augmented user message: retrieved context + original question
        augmented_message = USER_PROMPT.format(context=context, question=user_input)
        conversation.add_message(Message(Role.USER, augmented_message))

        # Send the full conversation history to the LLM and get its reply
        response = chat_client.get_completion(conversation.get_messages())
        # Append the assistant reply to history so follow-up questions have full context
        conversation.add_message(response)

        print(f"\nAssistant: {response.content}\n")


if __name__ == "__main__":
    run_chat()