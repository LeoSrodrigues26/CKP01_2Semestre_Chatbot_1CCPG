"""
context_rot.py — Demonstração de "context rot"

Ideia do experimento
------------------------------------------------------------------
1. Simulamos uma conversa em que, logo no início, o usuário revela uma
   informação crítica: uma alergia alimentar ("tenho alergia a
   amendoim"). Em seguida, inserimos vários turnos de "conversa de
   enchimento" sobre nutrição (perguntas genéricas), para simular uma
   sessão longa.
2. No final, perguntamos: "Pode sugerir um lanche rápido para mim?" —
   uma pergunta que SÓ é respondida corretamente se o modelo ainda
   "lembra" da alergia mencionada lá no início.
3. Repetimos essa mesma pergunta final várias vezes, mas cada vez
   fornecendo ao modelo apenas os N tokens mais recentes do histórico
   (janelas menores = mais "esquecimento" forçado, simulando memórias
   TokenBuffer mais agressivas ou janelas de contexto menores).
4. Medimos, para cada janela: quantos tokens de histórico couberam,
   quantos turnos sobreviveram ao corte, e se a resposta do modelo
   ainda menciona/evita amendoim (proxy objetivo de "lembrança").

Isso produz uma evidência concreta e mensurável de degradação de
qualidade conforme o contexto é reduzido — o "context rot" pedido no
enunciado — em vez de uma alegação sem dados.
------------------------------------------------------------------
"""

from dataclasses import dataclass

import tiktoken

from app.chain import obter_llm

# Conversa simulada: turno 0 contém a informação crítica (alergia).
# Os turnos seguintes são "conversa de enchimento" mais longa e realista,
# para que existam janelas de tokens em que a informação crítica do
# início já tenha sido "empurrada para fora" da janela.

CONVERSA_SIMULADA = [
    ("human", "Oi! Tenho alergia a amendoim, minha nutricionista pediu "
              "pra eu evitar totalmente qualquer traço do alimento. Pode "
              "me ajudar com dicas de lanches para o meu dia a dia?"),
    ("ai", "Claro! Vou levar essa alergia a amendoim em conta em todas as "
           "sugestões daqui pra frente. Para eu te ajudar melhor, me conta: "
           "você prefere lanches doces ou salgados, e costuma levar comida "
           "pronta de casa ou compra no caminho para o trabalho?"),
    ("human", "Prefiro salgado, e gosto de coisas práticas pra levar pro "
              "trabalho, algo que não precise de geladeira o dia todo."),
    ("ai", "Entendido. Salgados práticos que não exigem refrigeração o dia "
           "inteiro incluem barrinhas de cereal sem oleaginosas, biscoitos "
           "integrais com queijo processado individual, ou mix de grão de "
           "bico assado com temperos. Todos são fáceis de carregar."),
    ("human", "Quais frutas têm mais fibra e ajudam a dar saciedade no meio da tarde?"),
    ("ai", "Maçã e pera com casca, frutas vermelhas como morango e amora, "
           "além de ameixa e goiaba, costumam ter bastante fibra solúvel e "
           "insolúvel, o que ajuda bastante na saciedade entre as refeições "
           "principais do dia."),
    ("human", "E sobre hidratação, quantos litros de água por dia são realmente recomendados?"),
    ("ai", "Em geral, recomenda-se entre 2 e 2,5 litros de água por dia para "
           "um adulto, mas isso varia bastante com o clima, o nível de "
           "atividade física e a composição corporal de cada pessoa. Em dias "
           "muito quentes ou de treino intenso, essa necessidade pode ser maior."),
    ("human", "O que você acha de granola no café da manhã, é uma boa escolha?"),
    ("ai", "Pode ser uma ótima opção, desde que você observe o rótulo com "
           "atenção: muitas granolas industrializadas têm bastante açúcar "
           "adicionado e óleo. Prefira versões com poucos ingredientes, "
           "aveia, castanhas variadas e sem adição de açúcar."),
    ("human", "Ovo cozido é uma boa fonte de proteína para o dia a dia?"),
    ("ai", "Sim, o ovo é uma fonte completa de proteína, com todos os "
           "aminoácidos essenciais, além de ser prático de preparar e "
           "transportar. Dois ovos cozidos já oferecem uma quantidade "
           "relevante de proteína para um lanche da manhã ou da tarde."),
    ("human", "Legal, e sobre carboidratos integrais, vale a pena trocar o pão branco pelo integral?"),
    ("ai", "Sim, geralmente vale a pena. Pães integrais têm mais fibras, "
           "vitaminas do complexo B e minerais, além de causar uma resposta "
           "glicêmica mais estável comparado ao pão branco refinado, o que "
           "ajuda a manter a saciedade por mais tempo."),
    ("human", "E quanto a temperos, existe algum que ajuda a reduzir a quantidade de sal usada nas receitas?"),
    ("ai", "Sim! Ervas frescas como alecrim, manjericão e orégano, além de "
           "alho, cebola, limão e pimenta-do-reino, ajudam bastante a dar "
           "sabor sem precisar aumentar o sal. Vinagre e ervas secas também "
           "são boas alternativas para realçar o sabor dos pratos."),
    ("human", "Faz sentido tomar café da manhã cedo mesmo sem fome logo ao acordar?"),
    ("ai", "Não existe uma regra rígida — o mais importante é respeitar os "
           "sinais de fome do seu corpo. Se você não sente fome logo ao "
           "acordar, pode esperar um pouco mais e comer sua primeira "
           "refeição quando sentir fome de verdade, desde que isso não vire "
           "hábito de pular refeições por muitas horas seguidas."),
]

PERGUNTA_FINAL = "Show, pode me sugerir um lanche rápido pra levar amanhã?"

# Janelas de tokens simuladas (do menor para o maior contexto). Vão de
# "lembra só o finalzinho da conversa" até "lembra a conversa quase inteira,
# incluindo o turno 0 onde a alergia foi mencionada".
JANELAS_TOKENS = [150, 350, 700, 1400]

PALAVRA_CHAVE_ALERGIA = "amendoim"


@dataclass
class ResultadoJanela:
    janela_tokens: int
    tokens_usados: int
    turnos_incluidos: int
    lembrou_alergia: bool
    resposta: str


def contar_tokens(texto: str, modelo_tiktoken: str = "cl100k_base") -> int:
    """Conta tokens usando tiktoken (aproximação — o gemma4 não tem
    tokenizer público no tiktoken, mas cl100k_base serve como proxy
    consistente para comparar as janelas entre si)."""
    encoder = tiktoken.get_encoding(modelo_tiktoken)
    return len(encoder.encode(texto))


def montar_historico_texto(turnos: list[tuple[str, str]]) -> str:
    linhas = []
    for papel, conteudo in turnos:
        prefixo = "Usuário" if papel == "human" else "NutriBot"
        linhas.append(f"{prefixo}: {conteudo}")
    return "\n".join(linhas)


def truncar_por_tokens(
    turnos: list[tuple[str, str]], limite_tokens: int
) -> list[tuple[str, str]]:
    """
    Mantém os turnos mais RECENTES que cabem no limite de tokens,
    descartando os mais antigos primeiro (comportamento típico de uma
    janela deslizante / TokenBufferMemory).
    """
    selecionados: list[tuple[str, str]] = []
    tokens_acumulados = 0

    for papel, conteudo in reversed(turnos):
        tokens_turno = contar_tokens(conteudo)
        if tokens_acumulados + tokens_turno > limite_tokens:
            break
        selecionados.insert(0, (papel, conteudo))
        tokens_acumulados += tokens_turno

    return selecionados


def rodar_experimento_context_rot() -> list[ResultadoJanela]:
    """
    Executa o experimento completo e retorna os resultados por janela.
    Requer OLLAMA_API_KEY configurada (chama o modelo de verdade).
    """
    llm = obter_llm(temperature=0.0)
    resultados: list[ResultadoJanela] = []

    for janela in JANELAS_TOKENS:
        turnos_truncados = truncar_por_tokens(CONVERSA_SIMULADA, janela)
        historico_texto = montar_historico_texto(turnos_truncados)

        prompt_final = (
            f"Histórico da conversa:\n{historico_texto}\n\n"
            f"Usuário: {PERGUNTA_FINAL}\nNutriBot:"
        )

        resposta = llm.invoke(prompt_final).content
        lembrou = PALAVRA_CHAVE_ALERGIA in resposta.lower() or "sem amendoim" in resposta.lower()

        resultados.append(
            ResultadoJanela(
                janela_tokens=janela,
                tokens_usados=contar_tokens(historico_texto),
                turnos_incluidos=len(turnos_truncados),
                lembrou_alergia=lembrou,
                resposta=resposta,
            )
        )

    return resultados


def imprimir_tabela(resultados: list[ResultadoJanela]) -> None:
    print(f"{'Janela (tokens)':<16}{'Tokens usados':<15}{'Turnos':<9}{'Lembrou alergia?':<18}")
    print("-" * 60)
    for r in resultados:
        print(
            f"{r.janela_tokens:<16}{r.tokens_usados:<15}{r.turnos_incluidos:<9}"
            f"{'SIM' if r.lembrou_alergia else 'NÃO':<18}"
        )


def gerar_grafico(resultados: list[ResultadoJanela], caminho_saida: str = "context_rot.png") -> None:
    """Diferencial: gráfico comparando janela de tokens x memória preservada."""
    import matplotlib.pyplot as plt

    janelas = [r.janela_tokens for r in resultados]
    lembrou = [1 if r.lembrou_alergia else 0 for r in resultados]

    plt.figure(figsize=(6, 4))
    plt.bar([str(j) for j in janelas], lembrou, color="#E6007A")
    plt.ylim(0, 1.2)
    plt.yticks([0, 1], ["Não lembrou", "Lembrou"])
    plt.xlabel("Janela de contexto (tokens)")
    plt.title("Context rot: retenção da alergia mencionada no início da conversa")
    plt.tight_layout()
    plt.savefig(caminho_saida)
    print(f"Gráfico salvo em {caminho_saida}")


if __name__ == "__main__":
    resultados = rodar_experimento_context_rot()
    imprimir_tabela(resultados)
    gerar_grafico(resultados)