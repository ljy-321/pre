"""
评测集与评测指标
"""

from typing import List, Dict
import json

class EvaluationDataset:
    """
    自建评测数据集
    三类问题：事实型(70%) / 对比型(20%) / 汇总型(10%)
    """
    
    @staticmethod
    def create_eval_set() -> List[Dict]:
        """创建评测集"""
        return [
            # 事实型 (70%)
            {
                "type": "fact",
                "query": "H100的功率是多少",
                "ground_truth_chunk_id": 5,
                "expected_answer": "H100功率约700W",
            },
            {
                "type": "fact",
                "query": "海上风机基础类型有哪些",
                "ground_truth_chunk_id": 12,
                "expected_answer": "包括单桩、吸力桶、重力式等",
            },
            {
                "type": "fact",
                "query": "风电机组的尺寸规格",
                "ground_truth_chunk_id": 8,
                "expected_answer": "功率越大，尺寸越大",
            },
            # 对比型 (20%)
            {
                "type": "comparison",
                "query": "直驱和双馈风机的区别",
                "ground_truth_chunk_id": 20,
                "expected_answer": "直驱效率高但成本高，双馈成熟但效率低",
            },
            {
                "type": "comparison",
                "query": "陆上风电和海上风电的优缺点对比",
                "ground_truth_chunk_id": 25,
                "expected_answer": "海上风资源好但成本高，陆上反之",
            },
            # 汇总型 (10%)
            {
                "type": "summary",
                "query": "2025年海上风电发展趋势",
                "ground_truth_chunk_id": 30,
                "expected_answer": "大功率、深远海、技术升级是主要方向",
            },
        ]
    
    @staticmethod
    def save_eval_set(eval_set: List[Dict], save_path: str):
        """保存评测集"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(eval_set, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_eval_set(save_path: str) -> List[Dict]:
        """加载评测集"""
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)


class EvaluationMetrics:
    """
    评测指标计算
    """
    
    @staticmethod
    def recall_at_k(predicted_ids: List[int], 
                   ground_truth_id: int,
                   k: int = 5) -> int:
        """
        Recall@K
        返回1如果正确答案在Top-K中，否则0
        """
        return 1 if ground_truth_id in predicted_ids[:k] else 0
    
    @staticmethod
    def mrr(predicted_ids: List[int], 
           ground_truth_id: int) -> float:
        """
        MRR (Mean Reciprocal Rank)
        返回倒数排名
        """
        try:
            rank = predicted_ids.index(ground_truth_id) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    @staticmethod
    def ndcg_at_k(predicted_scores: List[float],
                 ground_truth_positions: List[int],
                 k: int = 5) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)
        用于评估排序质量
        """
        # DCG: 累计折损增益
        dcg = 0.0
        for i, pos in enumerate(ground_truth_positions[:k]):
            gain = 1.0 if pos == 0 else 0.0
            discount = 1.0 / np.log2(i + 2)
            dcg += gain * discount
        
        # IDCG: 理想DCG（最优排序）
        idcg = 0.0
        for i in range(min(k, len(ground_truth_positions))):
            if i == 0:
                gain = 1.0
                discount = 1.0 / np.log2(i + 2)
                idcg += gain * discount
        
        return dcg / idcg if idcg > 0 else 0.0


class RAGEvaluation:
    """
    完整的RAG系统评测
    """
    
    def __init__(self, eval_set: List[Dict]):
        self.eval_set = eval_set
        self.results = []
    
    def evaluate(self, retrieval_fn, generation_fn) -> Dict:
        """
        评测完整的RAG系统
        
        Args:
            retrieval_fn: 检索函数，返回List[Dict]
            generation_fn: 生成函数，返回Dict
        
        Returns:
            评测结果汇总
        """
        recall_at_5_list = []
        recall_at_10_list = []
        accuracy_list = []
        
        for item in self.eval_set:
            query = item["query"]
            ground_truth_chunk_id = item["ground_truth_chunk_id"]
            
            # 检索
            retrieved = retrieval_fn(query)
            retrieved_ids = [r["chunk_id"] for r in retrieved]
            
            # 生成
            generated = generation_fn(query, retrieved)
            
            # 计算指标
            recall_at_5 = EvaluationMetrics.recall_at_k(
                retrieved_ids, ground_truth_chunk_
