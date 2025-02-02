import os

import numpy as np
import streamlit as st
import textract
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

st.set_page_config(
    page_title="Islamic Texts",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

WORKING_DIR = "./processed"
BOOKS_DIR = "./data_mini"
PROCESS_FLAG = os.path.join(WORKING_DIR, ".processed")

load_dotenv()


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


@st.cache_resource
def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768, max_token_size=2048, func=embedding_func
        ),
    )

    if os.path.exists(BOOKS_DIR):
        if not os.path.exists(PROCESS_FLAG):
            islamic_texts = process_documents_folder(BOOKS_DIR)
            if islamic_texts:
                rag.insert(islamic_texts)
                with open(PROCESS_FLAG, "w") as f:
                    f.write("processed")
                st.success("Successfully loaded Islamic texts into RAG!")
            else:
                st.error("No processable texts found.")
        else:
            st.info("Documents already processed.")
    else:
        st.error(f"Books directory '{BOOKS_DIR}' not found.")

    return rag


def process_documents_folder(folder_path):
    """Extract text from all supported files in a directory"""
    all_text = ""
    supported_extensions = (".pdf", ".txt")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                text_content = textract.process(file_path)
                all_text += text_content.decode("utf-8") + "\n\n"
                st.success(f"Successfully processed: {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
    return all_text


def main():
    st.markdown(
        "<h1 style='text-align: center;'>📚 Islamic Texts</h1>",
        unsafe_allow_html=True,
    )
    st.write("")
    rag = initialize_rag()

    with st.form("query_form"):
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask a question about Islamic texts...",
        )

        # mode = st.selectbox("Select RAG mode:", ["naive", "local", "global", "hybrid"])
        mode = "hybrid"

        submit_button = st.form_submit_button("Generate response")

        custom_prompt = """You are a helpful assistant that provides accurate, informative answers based on the given context.

Follow these guidelines:

1. If the question is COMPLETELY unrelated to the context (e.g., asking about mathematics when the context is about history), append "NO_RELEVANT_SOURCE" to your response.
2. If the question is even partially related to the topics in the context, provide an answer based on the available information WITHOUT adding "NO_RELEVANT_SOURCE".
3. If you're unsure about relevance, err on the side of providing an answer without the special token.

Provide a clear, direct answer based on relevant information from the context."""

    if submit_button and query:
        with st.spinner("Generating response..."):
            try:
                context = rag.query(
                    query, param=QueryParam(mode=mode, only_need_context=True)
                )
                answer = rag.query(query, param=QueryParam(mode=mode))
                st.write("#### Answer")
                st.write(answer)

                with st.expander("📑 Source Information", expanded=False):
                    if context:
                        st.info(
                            """
                            **Context**
                            """
                        )
                        st.code(context)

                    else:
                        st.warning("⚠️ No specific source found for this response.")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()
