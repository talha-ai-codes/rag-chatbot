"""
Takes retrieved chunks + the user's question, sends them to an LLM,
and returns a grounded answer.

Works with OpenAI directly, OR with free alternatives like Groq
(Groq uses the same OpenAI-style API, you just change base_url + model).
"""

from openai import OpenAI


def build_client(api_key: str, base_url: str | None = None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    if not chunks:
        return "I don't have enough information in the documents to answer that."

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers using the provided context from documents. "
        "If the answer is not in the context, say clearly that you don't know — "
        "never make up information. Keep answers concise. You may refer back to "
        "earlier parts of the conversation to understand follow-up questions."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation turns so follow-up questions make sense
    if chat_history:
        for turn in chat_history[-6:]:  # last 3 exchanges (user+assistant pairs)
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content
