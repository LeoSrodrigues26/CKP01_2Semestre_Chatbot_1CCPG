"""
chain.py — Pipeline LCEL do Assistente de Nutrição

Implementa a arquitetura de 2 chains ensinada na Aula 03:

1) Chain de conversa (chat livre) — ConversationChain + memória Buffer
   (ver memory_manager.py). Usada para a interação natural com o
   usuário no Gradio.

2) Chain de saída estruturada — pipeline LCEL puro com o operador `|`:
       ChatPromptTemplate | ChatOllama | PydanticOutputParser
   Usada para classificar/analisar a consulta em um objeto
   AnaliseConsulta (schemas.py), validado por Pydantic v2.

Modelo: exclusivamente gemma4:cloud via Ollama Cloud (obrigatório pelo
enunciado do CKP01). A chave é lida de OLLAMA_API_KEY (.env) — nunca
hardcoded no código.
"""

import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama

from app.memory_manager import criar_conversation_chain
from app.prompts import SYSTEM_PROMPT_ANALISE
from app.schemas import AnaliseConsulta

load_dotenv()

MODELO = "gemma4:cloud"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"


def obter_llm(temperature: float = 0.3) -> ChatOllama:
    """
    Cria e retorna o ChatOllama configurado para usar gemma4:cloud via
    Ollama Cloud, autenticado com OLLAMA_API_KEY (.env).
    """
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. Copie .env.example para .env "
            "e preencha sua chave (crie uma em https://ollama.com/settings/keys)."
        )

    return ChatOllama(
        model=MODELO,
        base_url=OLLAMA_CLOUD_BASE_URL,
        temperature=temperature,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"},
        },
    )


def construir_chain_chat(llm: ChatOllama | None = None) -> ConversationChain:
    """Chain 1/2: conversa livre com memória Buffer (Aula 03)."""
    llm = llm or obter_llm()
    return criar_conversation_chain(llm)


def construir_chain_analise(llm: ChatOllama | None = None) -> RunnableSerializable:
    """
    Chain 2/2: pipeline LCEL de saída estruturada.

    prompt | llm | parser  — usando o operador `|` exigido pelo CKP01,
    com PydanticOutputParser validando o resultado contra AnaliseConsulta.
    """
    llm = llm or obter_llm(temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=AnaliseConsulta)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_ANALISE),
            ("human", "Consulta do usuário:\n{consulta}\n\nHistórico relevante:\n{historico}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Pipeline LCEL — operador pipe obrigatório pelo enunciado.
    chain = prompt | llm | parser
    return chain


def analisar_consulta(consulta: str, historico: str = "") -> AnaliseConsulta:
    """Função utilitária: roda a chain 2 e retorna o objeto validado."""
    chain = construir_chain_analise()
    resultado: AnaliseConsulta = chain.invoke(
        {"consulta": consulta, "historico": historico}
    )
    return resultado