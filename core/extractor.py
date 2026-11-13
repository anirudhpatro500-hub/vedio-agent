from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import SecretStr

import os


def _get_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=SecretStr(api_key) if api_key is not None else None,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_completion_tokens=4096,
    )


def extract_action_items(transcript: str) -> str:
    llm = _get_llm()
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", "Extract all action items and open questions from this meeting transcript. "
                       "Return them as a bullet list. If none exist, say 'None.'"),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return chain.invoke(transcript[:4000])


def extract_key_decisions(transcript: str) -> str:
    llm = _get_llm()
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", "Extract all key decisions made in this meeting transcript. "
                       "Return them as a bullet list. If none exist, say 'None.'"),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return chain.invoke(transcript[:4000])   

def extract_questions(transcript: str) -> str:
    llm = _get_llm()
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", "Extract all open questions, unresolved issues, or topics "
                       "that were raised but not answered in this meeting transcript. "
                       "Return them as a bullet list. If none exist, say 'None.'"),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    return chain.invoke(transcript[:4000])   