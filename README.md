# Episodic Memory for GPT

Lucas Matias Caetano,  
"Memória Episódica para Sistemas Baseados em GPT: Uma Implementação Prática"

| DOWNLOADS DISPONÍVEIS |
| :------------------: |
| [DATASET](https://github.com/lucasmatias/Episodic_Memory_for_GPT/blob/main/memorias_teste.csv) |

## Índice

- [Descrição](#descrição)
- [Instalação Bibliotecas](#Instalação-bibliotecas)
- [Como Usar](#como-usar)
  - [Inicialização do Sistema](#inicialização-do-sistema)
  - [Armazenamento de Memórias](#armazenamento-de-memórias)
  - [Recuperação de Memórias](#recuperação-de-memórias)
  - [Geração de Contexto](#geração-de-contexto)
- [Avaliação](#avaliação)
- [Exemplos](#exemplos)
- [Detalhes Técnicos](#detalhes-técnicos)

## Descrição

Sistema de memória episódica para enriquecer modelos baseados em GPT com capacidade de armazenar e recuperar informações contextuais. Utiliza ChromaDB para armazenamento vetorial e Sentence Transformers para geração de embeddings.

Principais funcionalidades:
- Etiquetagem temporal automática
- Divisão de textos longos
- Recuperação por similaridade
- Integração com prompts GPT

## Instalação bibliotecas

```bash
pip install chromadb sentence-transformers pydantic
```

## Como Usar

### Inicialização do Sistema
```python
memory_system = EpisodicMemorySystem()
```

### Armazenamento de Memórias
```python
memory_ids = memory_system.store_memory(
    "O usuário gosta de música clássica e jazz.",
    {"tópico": "música"}
)
```

### Recuperação de Memórias
```python
memórias = memory_system.retrieve_memories(
    "Quais são os gostos musicais do usuário?",
    n_results=3
)
```

### Geração de Contexto
```python
prompt = "Descreva os interesses do usuário."
prompt_enriquecido = memory_system.generate_context_with_memories(prompt)
```

## Avaliação

Módulo de avaliação com métricas:

```python
evaluator = EpisodicMemoryEvaluator(memory_system)
evaluator.add_test_case(
    query="Quais são os interesses do usuário?",
    relevant_memory_ids=["id_memória"],
    description="Teste de preferências musicais"
)
resultados = evaluator.evaluate(k=3)
```

Métricas incluídas:

- Precisão@k
- Revocação@k
- MRR (Mean Reciprocal Rank)
- Similaridade média

## Exemplos

### Uso Básico
```python
# Inicializar
memory_system = EpisodicMemorySystem()

# Armazenar
memory_system.store_memory("O usuário adora pizza de calabresa.", {"tópico": "comida"})

# Recuperar
memórias = memory_system.retrieve_memories("Qual a comida favorita do usuário?")

# Contexto
enriquecido = memory_system.generate_context_with_memories("Fale sobre as preferências alimentares.")
```

## Detalhes Técnicos

Tecnologias utilizadas:

- **ChromaDB**: Armazenamento vetorial
- **Sentence Transformers**: Geração de embeddings
- **Metadados temporais**: Data/hora automáticos
- **Chunking**: Divisão de textos (padrão: 512 tokens)

## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.
