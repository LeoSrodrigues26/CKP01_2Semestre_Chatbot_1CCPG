"""
main.py — Entry point do Assistente de Nutrição

Roda: python -m app.main
Abre uma interface Gradio em http://localhost:7860 com duas abas:

1) Chat — conversa livre com memória Buffer (Chain 1/2 da Aula 03).
2) Análise estruturada — roda a Chain 2/2 (LCEL + Pydantic) sobre a
   última pergunta do usuário e mostra o AnaliseConsulta validado.
"""

import gradio as gr

from app.chain import analisar_consulta, construir_chain_chat

# Chain de conversa é criada uma vez e mantém a memória Buffer entre
# as mensagens da mesma sessão do Gradio.
chain_chat = construir_chain_chat()


def responder_chat(mensagem: str, historico_ui: list) -> tuple[str, list]:
    """Callback do componente de chat do Gradio."""
    resposta = chain_chat.predict(input=mensagem)
    historico_ui = historico_ui + [(mensagem, resposta)]
    return "", historico_ui


def rodar_analise(mensagem: str) -> str:
    """Callback do botão de análise estruturada (Chain 2/2)."""
    if not mensagem.strip():
        return "Digite uma consulta antes de analisar."

    try:
        analise = analisar_consulta(consulta=mensagem)
    except Exception as exc:  # noqa: BLE001 - mostra erro amigável na UI
        return f"Erro ao analisar consulta: {exc}"

    return analise.model_dump_json(indent=2)


def construir_interface() -> gr.Blocks:
    with gr.Blocks(title="NutriBot — Assistente de Nutrição") as demo:
        gr.Markdown("# 🥗 NutriBot — Assistente de Nutrição (CKP01)")
        gr.Markdown(
            "Converse livremente na aba **Chat** ou use **Análise "
            "estruturada** para ver a saída validada em Pydantic v2."
        )

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="NutriBot")
            entrada = gr.Textbox(
                label="Sua mensagem",
                placeholder="Ex: tenho intolerância a lactose, o que posso comer no café da manhã?",
            )
            entrada.submit(responder_chat, [entrada, chatbot], [entrada, chatbot])

        with gr.Tab("Análise estruturada"):
            entrada_analise = gr.Textbox(
                label="Consulta para analisar",
                placeholder="Ex: quero saber se posso comer banana antes de treinar",
            )
            botao = gr.Button("Analisar consulta")
            saida_json = gr.Code(label="AnaliseConsulta (JSON validado)", language="json")
            botao.click(rodar_analise, [entrada_analise], [saida_json])

    return demo


if __name__ == "__main__":
    interface = construir_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860)