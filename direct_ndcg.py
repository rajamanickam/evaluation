"""
Calculate True Retrieval Recall for RAG Systems

Retrieval Recall = (Number of relevant docs retrieved) / (Total number of relevant docs)
"""

from typing import List, Set, Dict
import numpy as np

class RetrievalEvaluator:
    """
    Evaluates retrieval performance including Recall, Precision, and MRR
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
        
        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score (0-3 or binary)
            k: Consider only top k documents (None = all)
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
        # Sort relevance scores in descending order
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
        
        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score
                             Can be binary (0/1) or graded (0-3)
            k: Calculate nDCG@k (None = all documents)
        
        Returns:
            nDCG score between 0 and 1 (1 = perfect ranking)
        """
        dcg = self.calculate_dcg(retrieved_ids, relevance_scores, k)
        idcg = self.calculate_idcg(relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(
        self, 
        query: str,
        retrieved_ids: List[str], 
        relevant_ids: List[str],
        relevance_scores: Dict[str, float] = None
    ) -> Dict:
        """
        Comprehensive evaluation for a single query
        
        Args:
            query: The search query
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: List of relevant document IDs (for binary metrics)
            relevance_scores: Optional dict of doc_id -> relevance score (for nDCG)
                            If not provided, binary relevance (0/1) is used
        """
        # If no relevance scores provided, create binary scores
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 if doc_id in relevant_ids else 0.0 
                              for doc_id in retrieved_ids}
        
        metrics = {
            "query": query,
            "recall": self.calculate_recall(retrieved_ids, relevant_ids),
            "precision": self.calculate_precision(retrieved_ids, relevant_ids),
            "f1": self.calculate_f1(retrieved_ids, relevant_ids),
            "recall@5": self.calculate_recall_at_k(retrieved_ids, relevant_ids, 5),
            "recall@10": self.calculate_recall_at_k(retrieved_ids, relevant_ids, 10),
            "mrr": self.calculate_mrr(retrieved_ids, relevant_ids),
            "ndcg": self.calculate_ndcg(retrieved_ids, relevance_scores),
            "ndcg@5": self.calculate_ndcg(retrieved_ids, relevance_scores, k=5),
            "ndcg@10": self.calculate_ndcg(retrieved_ids, relevance_scores, k=10),
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids),
            "num_relevant_retrieved": len(set(retrieved_ids).intersection(set(relevant_ids)))
        }
        
        self.results.append(metrics)
        return metrics
    
    def evaluate_dataset(
        self,
        queries: List[str],
        retrieved_ids_list: List[List[str]],
        relevant_ids_list: List[List[str]],
        relevance_scores_list: List[Dict[str, float]] = None
    ) -> Dict:
        """
        Evaluate multiple queries and return average metrics
        
        Args:
            queries: List of queries
            retrieved_ids_list: List of retrieved doc IDs for each query
            relevant_ids_list: List of relevant doc IDs for each query
            relevance_scores_list: Optional list of relevance score dicts for each query
        """
        self.results = []
        
        if relevance_scores_list is None:
            relevance_scores_list = [None] * len(queries)
        
        for query, retrieved, relevant, rel_scores in zip(
            queries, retrieved_ids_list, relevant_ids_list, relevance_scores_list
        ):
            self.evaluate_query(query, retrieved, relevant, rel_scores)
        
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
        
        return avg_metrics
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(self.results, 1):
            print(f"\nQuery {i}: {result['query']}")
            print(f"  Recall: {result['recall']:.4f} ({result['num_relevant_retrieved']}/{result['num_relevant']} relevant docs retrieved)")
            print(f"  Precision: {result['precision']:.4f} ({result['num_relevant_retrieved']}/{result['num_retrieved']} retrieved docs were relevant)")
            print(f"  F1 Score: {result['f1']:.4f}")
            print(f"  Recall@5: {result['recall@5']:.4f}")
            print(f"  MRR: {result['mrr']:.4f}")
            print(f"  nDCG: {result['ndcg']:.4f}")
            print(f"  nDCG@5: {result['ndcg@5']:.4f}")
            print(f"  nDCG@10: {result['ndcg@10']:.4f}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # Example: You have a knowledge base with document IDs
    # You need to manually label which documents are relevant for each query
    
    evaluator = RetrievalEvaluator()
    
    # Example 1: Single query evaluation
    query = "What is the capital of France?"
    
    # These are the document IDs your retriever returned (in order)
    retrieved_ids = ["doc_5", "doc_12", "doc_3", "doc_45", "doc_8"]
    
    # These are ALL the relevant document IDs in your knowledge base (ground truth)
    # You need to manually identify these or use human annotators
    relevant_ids = ["doc_5", "doc_12", "doc_23"]  # doc_23 exists but wasn't retrieved!
    
    result = evaluator.evaluate_query(query, retrieved_ids, relevant_ids)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved_ids)} documents: {retrieved_ids}")
    print(f"Relevant documents in KB: {relevant_ids}")
    print(f"\nRecall: {result['recall']:.4f}")  # 2/3 = 0.67 (missed doc_23)
    print(f"Precision: {result['precision']:.4f}")  # 2/5 = 0.40
    print(f"F1: {result['f1']:.4f}")
    print(f"nDCG: {result['ndcg']:.4f}")
    
    # Example 2: Using graded relevance scores for nDCG
    print("\n" + "="*80)
    print("EXAMPLE WITH GRADED RELEVANCE (0-3 scale)")
    print("="*80)
    
    query2 = "Machine learning algorithms"
    retrieved_ids2 = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant_ids2 = ["doc_A", "doc_B", "doc_C"]
    
    # Graded relevance: 3=highly relevant, 2=relevant, 1=somewhat relevant, 0=not relevant
    relevance_scores = {
        "doc_A": 3,  # Highly relevant (retrieved at rank 1) ✓
        "doc_B": 2,  # Relevant (retrieved at rank 2) ✓
        "doc_C": 1,  # Somewhat relevant (retrieved at rank 3) ✓
        "doc_D": 0,  # Not relevant (retrieved at rank 4) ✗
        "doc_E": 0,  # Not relevant (retrieved at rank 5) ✗
    }
    
    result2 = evaluator.evaluate_query(query2, retrieved_ids2, relevant_ids2, relevance_scores)
    
    print(f"\nQuery: {query2}")
    print(f"Retrieved: {retrieved_ids2}")
    print(f"Relevance scores: {relevance_scores}")
    print(f"\nRecall: {result2['recall']:.4f}")
    print(f"Precision: {result2['precision']:.4f}")
    print(f"nDCG: {result2['ndcg']:.4f}")  # Penalizes irrelevant docs at top ranks
    print(f"nDCG@3: {result2['ndcg@5']:.4f}")
    
    # Example 3: Evaluate multiple queries
    print("\n" + "="*80)
    print("EVALUATING MULTIPLE QUERIES")
    print("="*80)
    
    queries = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is photosynthesis?"
    ]
    
    # What your system retrieved for each query
    retrieved_ids_list = [
        ["doc_5", "doc_12", "doc_3", "doc_45"],
        ["doc_20", "doc_21", "doc_100"],
        ["doc_30", "doc_31", "doc_32", "doc_33", "doc_34"]
    ]
    
    # Ground truth: all relevant docs in your KB for each query
    # This requires manual annotation!
    relevant_ids_list = [
        ["doc_5", "doc_12", "doc_23"],  # 3 relevant docs exist
        ["doc_20", "doc_21", "doc_22"],  # 3 relevant docs exist
        ["doc_30", "doc_31", "doc_35"]   # 3 relevant docs exist
    ]
    
    avg_metrics = evaluator.evaluate_dataset(queries, retrieved_ids_list, relevant_ids_list)
    evaluator.print_results()
    
    print("\n" + "="*80)
    print("AVERAGE METRICS ACROSS ALL QUERIES")
    print("="*80)
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")