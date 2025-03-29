# -*- coding: utf-8 -*-

import datetime
import chromadb
import uuid
import numpy as np
import csv
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from IPython.display import display, Markdown

# Função para exibir resultados formatados no Colab
def display_results(title, content):
    display(Markdown(f"### {title}"))
    display(Markdown(content))

class EpisodicMemorySystem:
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """Inicializa o sistema de memória episódica"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # Configura o ChromaDB
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_temporal_tags(self) -> Dict[str, str]:
        """Gera metadados temporais"""
        now = datetime.datetime.now()
        return {
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "time_of_day": self._get_time_of_day(now)
        }

    def _get_time_of_day(self, time: datetime.datetime) -> str:
        """Determina o período do dia"""
        hour = time.hour
        if 5 <= hour < 12: return "morning"
        elif 12 <= hour < 18: return "afternoon"
        elif 18 <= hour < 22: return "evening"
        else: return "night"

    def _chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Divide o texto em chunks"""
        words = text.split()
        chunks, current_chunk = [], []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
            current_chunk.append(word)
            current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def store_memory(self, text: str, additional_metadata: Optional[Dict] = None) -> List[str]:
        """Armazena uma nova memória com metadados completos"""
        metadata = {
            **self._get_temporal_tags(),
            **(additional_metadata or {})
        }

        chunks = self._chunk_text(text)
        if not chunks:
            return []

        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Garante que cada chunk tenha seus próprios metadados completos
        metadatas = []
        for chunk_id, chunk in zip(ids, chunks):
            metadatas.append({
                **metadata,
                "chunk_id": chunk_id,
                "content": chunk[:100]  # Armazena um trecho para debug
            })

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return ids
    def generate_context_with_memories(self, current_prompt: str, n_memories: int = 3) -> str:
        """
        Gera um contexto enriquecido com memórias relevantes.

        Args:
            current_prompt: Prompt atual
            n_memories: Número de memórias a incluir

        Returns:
            String com o prompt enriquecido pelas memórias
        """
        # Recupera memórias relevantes
        memories = self.retrieve_memories(current_prompt, n_results=n_memories)

        if not memories:
            return current_prompt

        # Formata as memórias para inclusão no contexto
        memory_context = "\n\nMemórias relevantes:\n"
        for i, memory in enumerate(memories, 1):
            memory_context += f"{i}. {memory['content']} (Data: {memory['metadata']['date']}, Hora: {memory['metadata']['time']})\n"

        return current_prompt + memory_context

    def retrieve_memories(self, query: str, n_results: int = 3, min_similarity: float = 0.5) -> List[Dict]:
        """Recupera memórias com filtro de similaridade mínima"""
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results*2,  # Recupera mais para filtrar depois
            include=["documents", "metadatas", "distances"]
        )

        # Formata e filtra os resultados
        memories = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 - dist
            if similarity >= min_similarity:
                memories.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity
                })

        return memories[:n_results]  # Retorna apenas o número solicitado

class EpisodicMemoryEvaluator:
    def __init__(self, memory_system: EpisodicMemorySystem):
        self.memory_system = memory_system
        self.test_cases = []

    def add_test_case(self, query: str, relevant_memory_ids: List[str], description: str = ""):
        """Adiciona um caso de teste para avaliação"""
        self.test_cases.append({
            "query": query,
            "relevant_ids": relevant_memory_ids,
            "description": description
        })

    def _calculate_precision_recall(self, retrieved_ids: List[str], relevant_ids: List[str]) -> tuple:
        """Calcula precisão e recall para um caso"""
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
        return precision, recall

    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calcula o Mean Reciprocal Rank"""
        for rank, mem_id in enumerate(retrieved_ids, 1):
            if mem_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def evaluate(self, k: int = 3, verbose: bool = False) -> Dict[str, float]:
        """Executa a avaliação completa"""
        precision_scores = []
        recall_scores = []
        mrr_scores = []
        avg_similarity_scores = []

        for test_case in self.test_cases:
            memories = self.memory_system.retrieve_memories(test_case["query"], n_results=k)
            retrieved_ids = [mem['metadata']['chunk_id'] for mem in memories]

            if verbose:
                print(f"\nQuery: {test_case['query']}")
                print(f"Expected IDs: {test_case['relevant_ids']}")
                print(f"Retrieved IDs: {retrieved_ids}")
                for mem in memories:
                    print(f"- {mem['metadata']['chunk_id']} (sim: {mem['similarity']:.2f}): {mem['content'][:50]}...")

            precision, recall = self._calculate_precision_recall(retrieved_ids, test_case["relevant_ids"])
            mrr = self._calculate_mrr(retrieved_ids, test_case["relevant_ids"])
            avg_sim = np.mean([mem['similarity'] for mem in memories]) if memories else 0

            precision_scores.append(precision)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            avg_similarity_scores.append(avg_sim)

        return {
            "precision@k": np.mean(precision_scores),
            "recall@k": np.mean(recall_scores),
            "mrr": np.mean(mrr_scores),
            "avg_similarity": np.mean(avg_similarity_scores),
            "num_test_cases": len(self.test_cases)
        }

# Funções auxiliares para carregar dados
def load_test_memories(memory_system, csv_file: str) -> Dict[str, str]:
    """Carrega memórias de teste e retorna mapeamento de IDs"""
    memory_ids_map = {}

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            metadata = {
                "topic": row["topic"],
                "time_of_day": row["time_of_day"],
                "date": row["date"],
                "is_priority": bool(int(row["is_priority"]))
            }
            generated_ids = memory_system.store_memory(row["text"], metadata)
            memory_ids_map[row["id"]] = generated_ids[0]  # Assume 1 chunk por memória

    return memory_ids_map

def load_test_queries(csv_file: str, memory_ids_map: Dict[str, str]):
    """Carrega consultas de teste usando mapeamento de IDs"""
    test_cases = []

    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            generated_id = memory_ids_map.get(row["id"])
            if generated_id:
                test_cases.append({
                    "query": row["expected_query"],
                    "relevant_memory_ids": [generated_id],
                    "description": f"Teste para memória {row['id']} sobre {row['topic']}"
                })

    return test_cases

# Exemplo de uso no Colab
if __name__ == "__main__":
    # Inicializa o sistema
    display(Markdown("# Sistema de Memória Episódica no Colab"))
    memory_system = EpisodicMemorySystem()

    # Armazena algumas memórias de exemplo
    display(Markdown("## Armazenando memórias..."))

    # 1. Carregar memórias e obter mapeamento de IDs
    memory_ids_map = load_test_memories(memory_system, "memorias_teste.csv")
    display_results(f"{len(memory_ids_map)} Memórias armazenadas", "Conteúdos diversos")

    # 2. Carregar consultas de teste usando o mapeamento
    test_cases = load_test_queries("memorias_teste.csv", memory_ids_map)

    # 3. Configurar avaliador
    evaluator = EpisodicMemoryEvaluator(memory_system)
    for test in test_cases:
        evaluator.add_test_case(
            query=test["query"],
            relevant_memory_ids=test["relevant_memory_ids"],  # Change 'relevant_memory_ids' to 'relevant_ids'
            description=test["description"]
        )

    # 4. Executar avaliação
    results = evaluator.evaluate(k=3)
    display_results(f"Desempenho",
                    f"**Precision@k:** {results['precision@k']:.2f}  \n"
                    f"**Recall@k:** {results['recall@k']:.2f}  \n"
                    f"**MRR:** {results['mrr']:.2f}  \n"
                    f"**AVG Similarity:** {results['avg_similarity']:.2f}  \n"
                    f"**Qtd. de Casos de Teste:** {results['num_test_cases']}")

    # Simula uma consulta
    display(Markdown("## Recuperando memórias..."))
    query = "Quais são os interesses do usuário?"
    memories = memory_system.retrieve_memories(query, n_results=2)

    display_results(f"Memórias recuperadas para: '{query}'", "")
    for i, mem in enumerate(memories, 1):
        display_results(f"Memória {i}",
                       f"**Conteúdo:** {mem['content']}  \n"
                       f"**Data:** {mem['metadata']['date']}  \n"
                       f"**Similaridade:** {mem['similarity']:.2f}")

    # Demonstra o prompt enriquecido
    display(Markdown("## Contexto enriquecido com memórias"))
    current_prompt = "Com base nas informações conhecidas, descreva os interesses do usuário."
    enriched_prompt = memory_system.generate_context_with_memories(current_prompt)

    display_results("Prompt final", enriched_prompt)

# 1. Inicialize o sistema
memory_system = EpisodicMemorySystem()

# 2. Armazene memórias
memory_system.store_memory("O usuário gosta de música clássica e jazz.", {"topic": "música"})

# 3. Recupere memórias
memories = memory_system.retrieve_memories("Quais são os gostos musicais do usuário?")

# 4. Visualize os resultados
for mem in memories:
    print(f"Memória: {mem['content']}")
    print(f"Data: {mem['metadata']['date']}")
    print("---")

# 5. Gere contexto enriquecido
prompt = "Fale sobre as preferências musicais do usuário."
enriched = memory_system.generate_context_with_memories(prompt)
display(Markdown(enriched))