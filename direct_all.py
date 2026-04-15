"""
Calculate Retrieval and Generation Metrics for RAG Systems

Retrieval Metrics:
- Recall = (Number of relevant docs retrieved) / (Total number of relevant docs)
- Precision, F1, MRR, nDCG

Generation Metrics:
- Faithfulness = Are generated answers grounded in retrieved documents?
"""

from typing import List, Set, Dict, Optional
import numpy as np
import re

class RetrievalEvaluator:
    """
    Evaluates retrieval performance including Recall, Precision, MRR, and nDCG
    """
    
    def __init__(self):
        self.results = []
    
    def calculate_recall(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Recall: How many of the relevant docs did we retrieve?
        
        Args:
            retrieved_ids: IDs of documents retrieved by the system
            relevant_ids: IDs of all relevant documents (ground truth)
        
        Returns:
            Recall score between 0 and 1
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        recall = true_positives / len(relevant_set)
        
        return recall
    
    def calculate_precision(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Precision: Of what we retrieved, how many were relevant?
        """
        if len(retrieved_ids) == 0:
            return 0.0
        
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        precision = true_positives / len(retrieved_set)
        
        return precision
    
    def calculate_f1(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate F1 Score: Harmonic mean of Precision and Recall
        """
        precision = self.calculate_precision(retrieved_ids, relevant_ids)
        recall = self.calculate_recall(retrieved_ids, relevant_ids)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Calculate Recall@K: Recall considering only top K retrieved documents
        """
        top_k_retrieved = retrieved_ids[:k]
        return self.calculate_recall(top_k_retrieved, relevant_ids)
    
    def calculate_mrr(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank: 1 / rank of first relevant document
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    def calculate_dcg(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int = None
    ) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG)
        
        DCG = sum(rel_i / log2(i + 1)) for i in [1, k]
        """
        if k:
            retrieved_ids = retrieved_ids[:k]
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids, 1):
            relevance = relevance_scores.get(doc_id, 0)
            dcg += relevance / np.log2(i + 1)
        
        return dcg
    
    def calculate_idcg(
        self,
        relevance_scores: Dict[str, float],
        k: int = None
    ) -> float:
        """
        Calculate Ideal DCG (best possible ranking)
        """
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)
        
        if k:
            sorted_relevances = sorted_relevances[:k]
        
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances, 1):
            idcg += relevance / np.log2(i + 1)
        
        return idcg
    
    def calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG)
        
        nDCG = DCG / IDCG
        """
        dcg = self.calculate_dcg(retrieved_ids, relevance_scores, k)
        idcg = self.calculate_idcg(relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


class FaithfulnessEvaluator:
    """
    Evaluates faithfulness: whether generated answers are grounded in retrieved documents
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: If True, can use LLM for evaluation (more accurate but requires API)
                     If False, uses simpler heuristic methods
        """
        self.use_llm = use_llm
        self.results = []
    
    def calculate_token_overlap_faithfulness(
        self,
        generated_answer: str,
        retrieved_contexts: List[str]
    ) -> float:
        """
        Simple token overlap-based faithfulness score
        
        Measures what percentage of content words in the answer appear in contexts
        """
        # Simple tokenization (split on whitespace and punctuation)
        def tokenize(text):
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'is', 'was', 'are', 'were', 'be', 'been',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                         'could', 'should', 'it', 'its', 'this', 'that', 'these', 'those'}
            return [t for t in tokens if t not in stop_words and len(t) > 2]
        
        answer_tokens = set(tokenize(generated_answer))
        if len(answer_tokens) == 0:
            return 0.0
        
        # Combine all contexts
        combined_context = " ".join(retrieved_contexts)
        context_tokens = set(tokenize(combined_context))
        
        # Calculate overlap
        overlap = len(answer_tokens.intersection(context_tokens))
        faithfulness = overlap / len(answer_tokens)
        
        return faithfulness
    
    def calculate_sentence_faithfulness(
        self,
        generated_answer: str,
        retrieved_contexts: List[str],
        threshold: float = 0.5
    ) -> Dict:
        """
        Sentence-level faithfulness analysis
        
        Checks each sentence in the answer against contexts
        
        Returns:
            Dict with overall score and per-sentence details
        """
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', generated_answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return {"score": 0.0, "faithful_sentences": 0, "total_sentences": 0, "details": []}
        
        faithful_count = 0
        details = []
        
        for sentence in sentences:
            overlap = self.calculate_token_overlap_faithfulness(sentence, retrieved_contexts)
            is_faithful = overlap >= threshold
            faithful_count += int(is_faithful)
            
            details.append({
                "sentence": sentence,
                "overlap_score": overlap,
                "is_faithful": is_faithful
            })
        
        return {
            "score": faithful_count / len(sentences),
            "faithful_sentences": faithful_count,
            "total_sentences": len(sentences),
            "details": details
        }
    
    def calculate_faithfulness_with_claims(
        self,
        generated_answer: str,
        retrieved_contexts: List[str],
        claims: Optional[List[str]] = None
    ) -> Dict:
        """
        Claim-level faithfulness evaluation
        
        Args:
            generated_answer: The generated text
            retrieved_contexts: List of retrieved document texts
            claims: Optional list of atomic claims extracted from answer
                   If None, uses sentences as proxy for claims
        
        Returns:
            Dict with faithfulness metrics
        """
        if claims is None:
            # Use sentences as claims
            claims = re.split(r'[.!?]+', generated_answer)
            claims = [c.strip() for c in claims if c.strip()]
        
        if len(claims) == 0:
            return {
                "faithfulness": 0.0,
                "supported_claims": 0,
                "total_claims": 0,
                "unsupported_claims": []
            }
        
        supported = 0
        unsupported_claims = []
        
        for claim in claims:
            # Check if claim is supported by any context
            overlap = self.calculate_token_overlap_faithfulness(claim, retrieved_contexts)
            
            if overlap >= 0.5:  # Threshold for support
                supported += 1
            else:
                unsupported_claims.append(claim)
        
        return {
            "faithfulness": supported / len(claims),
            "supported_claims": supported,
            "total_claims": len(claims),
            "unsupported_claims": unsupported_claims
        }


class RAGEvaluator:
    """
    Comprehensive RAG evaluation combining retrieval and generation metrics
    """
    
    def __init__(self, use_llm_for_faithfulness: bool = False):
        self.retrieval_eval = RetrievalEvaluator()
        self.faithfulness_eval = FaithfulnessEvaluator(use_llm=use_llm_for_faithfulness)
        self.results = []
    
    def evaluate_query(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        generated_answer: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Comprehensive evaluation for a single query
        
        Args:
            query: The search query
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)
            generated_answer: Generated answer text (for faithfulness)
            retrieved_contexts: Texts of retrieved documents (for faithfulness)
            relevance_scores: Optional graded relevance scores
        """
        # Retrieval metrics
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 if doc_id in relevant_ids else 0.0 
                              for doc_id in retrieved_ids}
        
        metrics = {
            "query": query,
            "recall": self.retrieval_eval.calculate_recall(retrieved_ids, relevant_ids),
            "precision": self.retrieval_eval.calculate_precision(retrieved_ids, relevant_ids),
            "f1": self.retrieval_eval.calculate_f1(retrieved_ids, relevant_ids),
            "recall@5": self.retrieval_eval.calculate_recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall@10": self.retrieval_eval.calculate_recall_at_k(retrieved_ids, relevant_ids, 10),
            "mrr": self.retrieval_eval.calculate_mrr(retrieved_ids, relevant_ids),
            "ndcg": self.retrieval_eval.calculate_ndcg(retrieved_ids, relevance_scores),
            "ndcg@5": self.retrieval_eval.calculate_ndcg(retrieved_ids, relevance_scores, k=5),
            "ndcg@10": self.retrieval_eval.calculate_ndcg(retrieved_ids, relevance_scores, k=10),
        }
        
        # Faithfulness metrics (if answer and contexts provided)
        if generated_answer and retrieved_contexts:
            faithfulness_result = self.faithfulness_eval.calculate_faithfulness_with_claims(
                generated_answer, retrieved_contexts
            )
            metrics["faithfulness"] = faithfulness_result["faithfulness"]
            metrics["supported_claims"] = faithfulness_result["supported_claims"]
            metrics["total_claims"] = faithfulness_result["total_claims"]
            metrics["generated_answer"] = generated_answer
        
        self.results.append(metrics)
        return metrics
    
    def evaluate_dataset(
        self,
        queries: List[str],
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]],
        generated_answers: Optional[List[str]] = None,
        retrieved_contexts_list: Optional[List[List[str]]] = None,
        relevance_scores_list: Optional[List[Dict[str, float]]] = None
    ) -> Dict:
        """
        Evaluate multiple queries and return average metrics
        """
        self.results = []
        
        if relevance_scores_list is None:
            relevance_scores_list = [None] * len(queries)
        
        if generated_answers is None:
            generated_answers = [None] * len(queries)
        
        if retrieved_contexts_list is None:
            retrieved_contexts_list = [None] * len(queries)
        
        for query, retrieved, relevant, answer, contexts, rel_scores in zip(
            queries, retrieved_ids_list, relevant_ids_list, 
            generated_answers, retrieved_contexts_list, relevance_scores_list
        ):
            self.evaluate_query(query, retrieved, relevant, answer, contexts, rel_scores)
        
        # Calculate averages
        avg_metrics = {
            "avg_recall": np.mean([r["recall"] for r in self.results]),
            "avg_precision": np.mean([r["precision"] for r in self.results]),
            "avg_f1": np.mean([r["f1"] for r in self.results]),
            "avg_recall@5": np.mean([r["recall@5"] for r in self.results]),
            "avg_recall@10": np.mean([r["recall@10"] for r in self.results]),
            "avg_mrr": np.mean([r["mrr"] for r in self.results]),
            "avg_ndcg": np.mean([r["ndcg"] for r in self.results]),
            "avg_ndcg@5": np.mean([r["ndcg@5"] for r in self.results]),
            "avg_ndcg@10": np.mean([r["ndcg@10"] for r in self.results]),
        }
        
        # Add faithfulness if available
        faithfulness_scores = [r.get("faithfulness") for r in self.results if "faithfulness" in r]
        if faithfulness_scores:
            avg_metrics["avg_faithfulness"] = np.mean(faithfulness_scores)
        
        return avg_metrics
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "="*80)
        print("RAG EVALUATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            print(f"\nQuery {i}: {result['query']}")
            print(f"  Recall: {result['recall']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  F1 Score: {result['f1']:.4f}")
            print(f"  MRR: {result['mrr']:.4f}")
            print(f"  nDCG: {result['ndcg']:.4f}")
            
            if "faithfulness" in result:
                print(f"  Faithfulness: {result['faithfulness']:.4f} ({result['supported_claims']}/{result['total_claims']} claims supported)")
                if result.get("generated_answer"):
                    print(f"  Generated Answer: {result['generated_answer'][:100]}...")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    evaluator = RAGEvaluator()
    
    # Example 1: Evaluate with faithfulness
    print("="*80)
    print("EXAMPLE: RAG EVALUATION WITH FAITHFULNESS")
    print("="*80)
    
    query = "What is the capital of France?"
    retrieved_ids = ["doc_5", "doc_12", "doc_3"]
    relevant_ids = ["doc_5", "doc_12", "doc_23"]
    
    # Simulated retrieved documents
    retrieved_contexts = [
        "Paris is the capital and largest city of France.",
        "France is located in Western Europe. Paris has a population of 2.2 million.",
        "The Eiffel Tower is located in Paris."
    ]
    
    # Generated answer by LLM
    generated_answer = "The capital of France is Paris. It is the largest city in France with over 2 million people."
    
    result = evaluator.evaluate_query(
        query=query,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        generated_answer=generated_answer,
        retrieved_contexts=retrieved_contexts
    )
    
    print(f"\nQuery: {query}")
    print(f"Generated Answer: {generated_answer}")
    print(f"\nRetrieval Metrics:")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  nDCG: {result['ndcg']:.4f}")
    print(f"\nGeneration Metrics:")
    print(f"  Faithfulness: {result['faithfulness']:.4f}")
    print(f"  Supported Claims: {result['supported_claims']}/{result['total_claims']}")
    
    # Example 2: Unfaithful answer (hallucination)
    print("\n" + "="*80)
    print("EXAMPLE: DETECTING HALLUCINATIONS")
    print("="*80)
    
    query2 = "Who invented the telephone?"
    retrieved_ids2 = ["doc_20", "doc_21"]
    relevant_ids2 = ["doc_20", "doc_21"]
    
    retrieved_contexts2 = [
        "Alexander Graham Bell is credited with inventing the telephone in 1876.",
        "The telephone revolutionized long-distance communication."
    ]
    
    # Answer with hallucination
    hallucinated_answer = "Alexander Graham Bell invented the telephone in 1876. He also invented the radio and the television."
    
    result2 = evaluator.evaluate_query(
        query=query2,
        retrieved_ids=retrieved_ids2,
        relevant_ids=relevant_ids2,
        generated_answer=hallucinated_answer,
        retrieved_contexts=retrieved_contexts2
    )
    
    print(f"\nQuery: {query2}")
    print(f"Generated Answer: {hallucinated_answer}")
    print(f"\nFaithfulness: {result2['faithfulness']:.4f}")
    print(f"Note: Lower score indicates hallucinations (claims about radio/TV not in context)")
    
    # Example 3: Multiple queries
    print("\n" + "="*80)
    print("EVALUATING MULTIPLE QUERIES")
    print("="*80)
    
    queries = [
        "What is the capital of France?",
        "Who invented the telephone?"
    ]
    
    retrieved_ids_list = [
        ["doc_5", "doc_12", "doc_3"],
        ["doc_20", "doc_21"]
    ]
    
    relevant_ids_list = [
        ["doc_5", "doc_12", "doc_23"],
        ["doc_20", "doc_21"]
    ]
    
    generated_answers = [
        "The capital of France is Paris.",
        "Alexander Graham Bell invented the telephone in 1876."
    ]
    
    retrieved_contexts_list = [
        ["Paris is the capital of France.", "France is in Europe.", "Paris has the Eiffel Tower."],
        ["Alexander Graham Bell invented the telephone in 1876.", "The telephone revolutionized communication."]
    ]
    
    avg_metrics = evaluator.evaluate_dataset(
        queries=queries,
        retrieved_ids_list=retrieved_ids_list,
        relevant_ids_list=relevant_ids_list,
        generated_answers=generated_answers,
        retrieved_contexts_list=retrieved_contexts_list
    )
    
    evaluator.print_results()
    
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS ALL QUERIES")
    print("="*80)
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")