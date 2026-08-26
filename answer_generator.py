"""
Takes retrieved chunks + the user's question, sends them to an LLM,
and returns a grounded answer — either all at once, or streamed
token-by-token for a live "typing" effect in the UI.

Works with OpenAI directly, OR with free alternatives like Groq
(Groq uses the same OpenAI-style API, you just change base_url + model).
"""

from openai import OpenAI


def build_client(api_key: str, base_url: str | None = None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _build_messages(question: str, chunks: list[dict], chat_history: list[dict] | None) -> list[dict]:
    context_text = ""
    if chunks:
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )

    system_prompt = (
        "You are a knowledgeable AI/ML/programming assistant for a student. "
        "You are given some CONTEXT retrieved from the student's own documents. "
        "Rules:\n"
        "1. If the context directly answers the question, answer from it and mention it's from their documents.\n"
        "2. If the context is empty, irrelevant, or only partially answers the question, "
        "use your own general knowledge to give a complete, correct, detailed answer — "
        "but clearly say something like 'This isn't in your documents, but here's what I know:' first.\n"
        "3. Never pretend general knowledge came from the documents.\n"
        "4. Keep answers clear and well-structured; use examples where helpful.\n"
        "You may also refer to earlier parts of the conversation for follow-up questions."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for turn in chat_history[-6:]:  # last 3 exchanges
            messages.append({"role": turn["role"], "content": turn["content"]})

    if context_text:
        user_prompt = f"Context from documents:\n{context_text}\n\nQuestion: {question}"
    else:
        user_prompt = f"Question: {question}\n\n(No relevant document context found.)"

    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """Non-streaming version — returns the full answer as one string."""
    messages = _build_messages(question, chunks, chat_history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_answer_stream(
    client: OpenAI,
    model: str,
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
):
    """Streaming version — yields small text pieces as they arrive,
    so the UI can show a live 'typing' effect (used with st.write_stream)."""
    messages = _build_messages(question, chunks, chat_history)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
