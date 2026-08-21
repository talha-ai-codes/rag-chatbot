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


def generate_answer(client: OpenAI, model: str, question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in the documents to answer that."

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    system_prompt = (
        "You are a helpful assistant that answers ONLY using the provided context. "
        "If the answer is not in the context, say clearly that you don't know — "
        "never make up information. Keep answers concise."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
