"""
memory_manager.py — Gerenciamento de memória da conversa

Justificativa da escolha (Buffer):
------------------------------------------------------------------
Optamos por ConversationBufferMemory (memória "Buffer" completa) para
o Assistente de Nutrição pelos seguintes motivos:

1. Sessões curtas por natureza: uma consulta típica de nutrição
   (ex.: planejar um lanche, tirar dúvida sobre um alimento) costuma
   durar poucos turnos (5-15), então o buffer não cresce a ponto de
   ficar caro em tokens.
2. Informação crítica não pode ser perdida: alergias, intolerâncias e
   condições de saúde mencionadas no início da conversa PRECISAM ser
   lembradas em todos os turnos seguintes — um resumo (Summary) corre
   o risco de generalizar ou "esquecer" um detalhe crítico como
   "sou alérgico a amendoim", o que é inaceitável neste domínio.
3. Custo de tokens controlado: como limitamos a demonstração de
   context rot a janelas de até ~2000 tokens, o custo do buffer
   completo permanece previsível e mensurável.

Trade-off aceito: em conversas muito longas (dezenas de turnos), o
buffer cresceria demais e custaria mais tokens por chamada. Para este
CKP, esse cenário é fora de escopo (chatbot de sessão única), mas fica
documentado aqui como limitação conhecida — no CKP02 (RAG), parte
dessa memória de longo prazo passará a vir da base vetorial.
------------------------------------------------------------------
"""

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel

from app.prompts import SYSTEM_PROMPT_CHAT


def criar_memoria_buffer() -> ConversationBufferMemory:
    """Cria a memória Buffer usada pela ConversationChain do chat."""
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
    )


def criar_conversation_chain(llm: BaseChatModel) -> ConversationChain:
    """
    Monta a 1ª das 2 chains da arquitetura da Aula 03: a
    ConversationChain com memória, responsável pelo bate-papo livre
    com o usuário (não é a chain de saída estruturada — essa fica em
    chain.py, na função construir_chain_analise).
    """
    memoria = criar_memoria_buffer()

    # A ConversationChain clássica usa um prompt interno simples, então
    # injetamos nossa persona/regras diretamente no prefixo do prompt.
    from langchain.prompts import PromptTemplate

    template = f"""{SYSTEM_PROMPT_CHAT}

Histórico da conversa até agora:
{{history}}

Usuário: {{input}}
NutriBot:"""

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    return ConversationChain(
        llm=llm,
        memory=memoria,
        prompt=prompt,
        verbose=False,
    )