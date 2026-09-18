# CKP01 — Chatbot Profissional · Assistente de Nutrição

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

**Integrantes:** [PREENCHER: Nome Completo (RM00000)] · [PREENCHER] · [PREENCHER]

**Peso: 25% · Apresentação: Aula 04 · Entrega: 23:55 do dia da Aula 05 (.zip via Teams — só o líder)**

## Domínio

**Assistente de Nutrição (NutriBot)** — um chatbot que dá orientação
alimentar geral, educativa e prática no dia a dia: sugestões de
lanches e refeições, informações nutricionais gerais, hidratação,
leitura de rótulos e hábitos alimentares saudáveis.

**Por que esse domínio:** é um caso de uso real e recorrente (muitas
pessoas buscam orientação alimentar básica no dia a dia, sem precisar
de uma consulta formal para dúvidas simples), tem informações
críticas de segurança que precisam ser lembradas ao longo da conversa
(alergias, intolerâncias), o que torna a escolha e justificativa da
memória (requisito deste CKP) especialmente relevante — e é um bom
domínio para evoluir: no CKP02 vira uma base de conhecimento
nutricional (RAG) e no CKP03 vira uma ferramenta (tool) de um agente
maior de saúde/bem-estar.

**Usuários-alvo:** pessoas leigas em nutrição que buscam orientação
inicial, prática e educativa sobre alimentação — não substitui
acompanhamento profissional (isso é reforçado explicitamente no
system prompt, ver `app/prompts.py`).

## Requisitos atendidos

| Requisito | Status | Implementação |
|---|---|---|
| Pipeline LCEL | ✅ | `chain.py` — `prompt \| llm \| PydanticOutputParser()` em `construir_chain_analise()` |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud (`base_url="https://ollama.com"` + header `Authorization: Bearer` com `OLLAMA_API_KEY` do `.env`) |
| 2 chains (Aula 03) | ✅ | `construir_chain_chat()` (ConversationChain + memória) e `construir_chain_analise()` (pipeline LCEL de saída estruturada) |
| Memória gerenciada | ✅ | `ConversationBufferMemory` em `memory_manager.py`, justificada abaixo |
| Pydantic v2 (≥4 campos) | ✅ | `AnaliseConsulta` com 6 campos tipados em `schemas.py`, com `field_validator` |
| Context rot | ✅ | `context_rot.py` — testa retenção de uma alergia mencionada no início da conversa em janelas de 200/500/1000/2000 tokens |
| System prompt com persona | ✅ | `prompts.py` — XML tagging (`<persona>`, `<dominio>`, `<regras>`, `<restricoes>`) |
| Domínio documentado | ✅ | Este README + `prompts.py` |
| Context engineering com métricas (diferencial) | ✅ | `context_rot.py` conta tokens com `tiktoken` e gera gráfico com `matplotlib` (`gerar_grafico()`) |

## Como executar (local — sem Colab)

```bash
cp .env.example .env      # edite com sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m app.main        # Gradio: http://localhost:7860
```

Para rodar o experimento de context rot separadamente:

```bash
python -m app.context_rot
```

**Como conseguir a `OLLAMA_API_KEY`:** crie uma conta em
[ollama.com](https://ollama.com), gere uma chave em
`ollama.com/settings/keys` e cole em `.env`. Não é preciso rodar o
Ollama localmente — o modelo `gemma4:cloud` roda 100% na nuvem da
Ollama; a chave apenas autentica as chamadas.

## Justificativa da memória

Escolhemos **ConversationBufferMemory** (memória completa, sem
resumo) pelos seguintes motivos:

1. **Sessões curtas por natureza** — uma consulta de nutrição típica
   dura poucos turnos, então o buffer não cresce a ponto de ficar caro.
2. **Informação crítica não pode ser perdida** — alergias e
   intolerâncias mencionadas no início da conversa precisam ser
   lembradas em todos os turnos seguintes. Um resumo (Summary) corre
   o risco de generalizar ou "esquecer" um detalhe como "alergia a
   amendoim", o que é inaceitável neste domínio.
3. **Custo controlado** — como o experimento de context rot já mostra
   que a informação começa a se perder acima de determinada janela,
   sabemos exatamente o limite prático do buffer neste domínio.

**Trade-off aceito:** em conversas muito longas, o buffer cresceria
demais em tokens. Fora de escopo para este CKP (chatbot de sessão
única); no CKP02, parte dessa memória de longo prazo passa a vir da
base vetorial (RAG).

## Seção "Context rot" — o que o experimento mostra

O script `app/context_rot.py`:

1. Simula uma conversa em que o usuário revela uma alergia a
   amendoim logo no primeiro turno, seguida de vários turnos de
   perguntas genéricas de nutrição.
2. Ao final, pergunta "pode sugerir um lanche rápido?" — pergunta que
   só é respondida com segurança se o modelo ainda lembra da alergia.
3. Repete essa pergunta fornecendo apenas os turnos mais recentes que
   cabem em janelas de 150 / 350 / 700 / 1400 tokens (contados com
   `tiktoken`), simulando memórias mais agressivas / janelas de
   contexto menores. A conversa inteira tem ~700 tokens, então as
   janelas menores (150 e 350) forçam o esquecimento do turno inicial
   (onde a alergia foi mencionada), enquanto as janelas maiores (700 e
   1400) conseguem preservá-lo — um gradiente real de degradação, não
   arbitrário.
4. Registra, para cada janela: tokens usados, turnos incluídos e se a
   resposta ainda menciona/evita amendoim — uma métrica objetiva de
   "lembrança", em vez de uma alegação sem dados.
5. Gera uma tabela no terminal e um gráfico (`context_rot.png`)
   comparando janela de contexto x retenção da informação crítica.

Rode `python -m app.context_rot` (com `.env` configurado) para ver os
resultados reais com `gemma4:cloud`.

## Estrutura do projeto

```
ckp01_nutricao/
├── app/
│   ├── __init__.py
│   ├── main.py            # Interface Gradio + entry point
│   ├── chain.py            # Pipeline LCEL + configuração do ChatOllama
│   ├── memory_manager.py   # Memória Buffer + ConversationChain
│   ├── schemas.py           # Pydantic v2 (AnaliseConsulta)
│   ├── context_rot.py       # Demonstração de degradação de contexto
│   └── prompts.py           # System prompts (persona, regras, restrições)
├── .env.example
├── requirements.txt
└── README.md
```

## Observações

- Nenhum modelo deprecated é usado — exclusivamente `gemma4:cloud`
  via Ollama Cloud, conforme exigido.
- Comentários em PT-BR ao longo do código explicando decisões de
  design.
- Este projeto é a base que evoluirá para RAG no CKP02 e para uma
  tool de agente no CKP03 — por isso o código já está modularizado
  (`chain.py`, `memory_manager.py`, `schemas.py` isolados).
