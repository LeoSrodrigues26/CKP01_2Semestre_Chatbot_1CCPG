"""
schemas.py — Modelos Pydantic v2 do domínio de Nutrição

Define a estrutura de saída validada que o pipeline LCEL produz ao
analisar uma consulta do usuário. Usado junto com PydanticOutputParser
em chain.py (método ensinado na Aula 03 — JsonOutputParser retornaria
um dict "cru", sem validação de tipos/regras).
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

CategoriaConsulta = Literal[
    "alimentacao",
    "exercicio",
    "suplementacao",
    "duvida_geral",
    "outro",
]


class AnaliseConsulta(BaseModel):
    """
    Representa a análise estruturada de uma pergunta feita ao
    assistente de nutrição. Usada para classificar e triar a consulta
    do usuário antes/depois da resposta em linguagem natural.
    """

    tema_principal: str = Field(
        ...,
        description="Resumo em poucas palavras do assunto da consulta do usuário.",
        min_length=3,
    )

    categoria: CategoriaConsulta = Field(
        ...,
        description=(
            "Categoria da consulta: alimentacao, exercicio, suplementacao, "
            "duvida_geral ou outro."
        ),
    )

    nivel_prioridade: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Prioridade de acompanhamento profissional, de 1 (baixa, "
            "curiosidade geral) a 5 (alta, sinais de risco à saúde)."
        ),
    )

    recomendacao_resumida: str = Field(
        ...,
        description="Recomendação geral e não-clínica em até 2 frases.",
        min_length=5,
    )

    requer_profissional: bool = Field(
        ...,
        description=(
            "True se a consulta indica necessidade de encaminhar a um "
            "nutricionista ou médico (ex.: restrições, condições de saúde)."
        ),
    )

    palavras_chave: List[str] = Field(
        default_factory=list,
        description="Lista de 2 a 5 palavras-chave extraídas da consulta.",
    )

    @field_validator("palavras_chave")
    @classmethod
    def limitar_palavras_chave(cls, valor: List[str]) -> List[str]:
        """Garante que a lista de palavras-chave não fique vazia nem enorme."""
        if len(valor) > 5:
            return valor[:5]
        return valor

    @field_validator("tema_principal", "recomendacao_resumida")
    @classmethod
    def sem_texto_vazio(cls, valor: str) -> str:
        """Evita que o modelo retorne strings vazias ou só com espaços."""
        texto = valor.strip()
        if not texto:
            raise ValueError("O campo não pode ser vazio.")
        return texto