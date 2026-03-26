"""
Query理解与同义词扩展
"""

from typing import List, Dict
import json

class SynonymExpander:
    """
    轻量级同义词扩展
    维护专业领域同义词表
    """
    
    # 能源装备领域同义词表
    SYNONYM_DICT = {
        "类型": ["形式", "种类", "分类", "类别"],
        "功率": ["额定功率", "容量", "出力", "装机容量"],
        "尺寸": ["长度", "规格", "参数", "尺寸"],
        "区别": ["差异", "对比", "不同点", "区别"],
        "材料": ["材质", "原料"],
        "效率": ["转换效率", "发电效率", "利用率"],
        "基础": ["地基", "基础设施"],
        "风电": ["风力发电", "风能"],
        "光伏": ["光伏发电", "太阳能"],
        "海上": ["近海", "近岸"],
    }
    
    def __init__(self, synonym_dict: Dict = None):
        """
        Args:
            synonym_dict: 自定义同义词表
        """
        self.synonym_dict = synonym_dict or self.SYNONYM_DICT
    
    def expand_query(self, query: str) -> List[str]:
        """
        扩展Query为多个变体
        
        Example:
            "海上风电基础类型"
            → [
                "海上风电基础类型",
                "海上风电基础形式",
                "海上风力发电基础类型",
            ]
        """
        expanded = [query]  # 原始query
        
        # 逐个关键词扩展
        for word, synonyms in self.synonym_dict.items():
            if word in query:
                for syn in synonyms:
                    new_query = query.replace(word, syn)
                    if new_query not in expanded:
                        expanded.append(new_query)
        
        return expanded[:3]  # 最多返回3个变体
    
    def add_synonym(self, word: str, synonyms: List[str]):
        """动态添加同义词"""
        self.synonym_dict[word] = synonyms
    
    def load_from_file(self, filepath: str):
        """从文件加载同义词表"""
        with open(filepath, "r", encoding="utf-8") as f:
            self.synonym_dict = json.load(f)


class QueryUnderstanding:
    """
    Query理解与预处理
    包括：分词、实体识别（简化版）、意图分类
    """
    
    def __init__(self):
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            print("jieba not installed")
    
    def tokenize(self, query: str) -> List[str]:
        """分词"""
        tokens = self.jieba.cut(query, cut_all=False)
        return list(tokens)
    
    def classify_intent(self, query: str) -> str:
        """
        简单意图分类
        - fact: 事实型("H100功率是多少")
        - comparison: 对比型("直驱和双馈的区别")
        - summary: 汇总型("海上风电2025趋势")
        """
        keywords = {
            "fact": ["是什么", "多少", "参数", "规格", "尺寸"],
            "comparison": ["区别", "对比", "不同", "vs", "vs"],
            "summary": ["趋势", "发展", "现状", "应用", "实践"],
        }
        
        for intent, kws in keywords.items():
            if any(kw in query for kw in kws):
                return intent
        
        return "fact"  # 默认事实型
    
    def preprocess(self, query: str) -> Dict:
        """
        完整预处理流程
        """
        return {
            "raw_query": query,
            "tokens": self.tokenize(query),
            "intent": self.classify_intent(query),
            "length": len(query),
        }


class QueryExpander:
    """
    综合的Query扩展器
    包括：同义词扩展 + 分词扩展
    """
    
    def __init__(self):
        self.synonym_expander = SynonymExpander()
        self.query_understanding = QueryUnderstanding()
    
    def expand(self, query: str) -> Dict:
        """
        返回扩展结果
        {
            "variants": [...],  # 扩展query列表
            "intent": "...",
            "analysis": {...}
        }
        """
        variants = self.synonym_expander.expand_query(query)
        analysis = self.query_understanding.preprocess(query)
        
        return {
            "original": query,
            "variants": variants,
            "intent": analysis["intent"],
            "analysis": analysis
        }


if __name__ == "__main__":
    expander = QueryExpander()
    
    # 测试1：同义词扩展
    query1 = "海上风电基础类型"
    result1 = expander.expand(query1)
    print(f"\n原始Query: {result1['original']}")
    print(f"扩展变体: {result1['variants']}")
    print(f"意图: {result1['intent']}")
    
    # 测试2
    query2 = "直驱和双馈的区别"
    result2 = expander.expand(query2)
    print(f"\n原始Query: {result2['original']}")
    print(f"扩展变体: {result2['variants']}")
    print(f"意图: {result2['intent']}")

      
"""
混合召回与融合策略 (RRF)
"""

from typing import List, Dict
import numpy as np

class ReciprocRankFusion:
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    
    公式: RRF_score(d) = Σ(1 / (k + rank_d))
    其中k=60为平滑因子
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(self, 
             dense_results: List[Dict],  # [{chunk_id, score, ...}, ...]
             bm25_results: List[Dict],
             top_k: int = 5) -> List[Dict]:
        """
        融合Dense召回和BM25召回
        
        Args:
            dense_results: Dense检索结果（带score）
            bm25_results: BM25检索结果（带score）
            top_k: 最终返回数量
        
        Returns:
            融合后的top-k结果，按RRF分数排序
        """
        # 构建chunk_id到rank的映射
        dense_ranks = {r["chunk_id"]: i for i, r in enumerate(dense_results)}
        bm25_ranks = {r["chunk_id"]: i for i, r in enumerate(bm25_results)}
        
        # 计算RRF分数
        all_chunk_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores = {}
        
        for chunk_id in all_chunk_ids:
            score = 0.0
            
            # Dense贡献
            if chunk_id in dense_ranks:
                score += 1.0 / (self.k + dense_ranks[chunk_id])
            
            # BM25贡献
            if chunk_id in bm25_ranks:
                score += 1.0 / (self.k + bm25_ranks[chunk_id])
            
            rrf_scores[chunk_id] = score
        
        # 排序取top-k
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 构建返回结果
        final_results = []
        for chunk_id, rrf_score in sorted_results:
            # 优先用dense结果的内容（通常更详细）
            if chunk_id in dense_ranks:
                result = dense_results[dense_ranks[chunk_id]].copy()
            else:
                result = bm25_results[bm25_ranks[chunk_id]].copy()
            
            result["rrf_score"] = rrf_score
            final_results.append(result)
        
        return final_results


class WeightedFusion:
    """
    加权融合策略
    RRF_score = α * dense_score + β * bm25_score
    """
    
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: Dense权重（通常0.5-0.7）
            beta: BM25权重（通常0.3-0.5）
        """
        assert alpha + beta == 1.0, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
    
    def normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """将分数归一化到[0,1]"""
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = []
        for r in results:
            r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score + 1e-8)
            normalized.append(r)
        
        return normalized
    
    def fuse(self, dense_results: List[Dict], bm25_results: List[Dict], 
             top_k: int = 5) -> List[Dict]:
        """融合"""
        # 归一化
        dense_normalized = self.normalize_scores(dense_results)
        bm25_normalized = self.normalize_scores(bm25_results)
        
        # 构建映射
        dense_map = {r["chunk_id"]: r for r in dense_normalized}
        bm25_map = {r["chunk_id"]: r for r in bm25_normalized}
        
        # 加权融合
        all_chunk_ids = set(dense_map.keys()) | set(bm25_map.keys())
        fused_scores = {}
        
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_map:
                score += self.alpha * dense_map[chunk_id]["normalized_score"]
            if chunk_id in bm25_map:
                score += self.beta * bm25_map[chunk_id]["normalized_score"]
            fused_scores[chunk_id] = score
        
        # 排序取top-k
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        final_results = []
        for chunk_id, fused_score in sorted_results:
            if chunk_id in dense_map:
                result = dense_map[chunk_id].copy()
            else:
                result = bm25_map[chunk_id].copy()
            result["fused_score"] = fused_score
            final_results.append(result)
        
        return final_results


class HybridRetrieval:
    """
    完整的混合检索系统
    """
    
    def __init__(self, embedding_index, bm25_index):
        """
        Args:
            embedding_index: FAISS索引
            bm25_index: BM25索引
        """
        self.embedding_index = embedding_index
        self.bm25_index = bm25_index
        self.fusion_strategy = ReciprocRankFusion()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        完整的混合检索流程
        1. Dense检索top_k * 2（多一些备选）
        2. BM25检索top_k * 2
        3. RRF融合得到最终top_k
        """
        # Query embedding
        # query_embedding = self.embedding_index.model.encode([query])
        query_embedding = np.random.randn(1, 1024).astype(np.float32)  # 模拟
        
        # Dense检索
        dense_results = self.embedding_index.search(
            query_embedding,
            k=top_k * 2
        )
        
        # BM25检索
        bm25_results = self.bm25_index.search(query, k=top_k * 2)
        
        # 融合
        final_results = self.fusion_strategy.fuse(
            dense_results,
            bm25_results,
            top_k=top_k
        )
        
        return final_results


if __name__ == "__main__":
    # 示例：演示融合效果
    dense_results = [
        {"chunk_id": 1, "score": 0.9, "content": "..."},
        {"chunk_id": 2, "score": 0.7, "content": "..."},
        {"chunk_id": 3, "score": 0.6, "content": "..."},
    ]
    
    bm25_results = [
        {"chunk_id": 2, "score": 0.95, "content": "..."},
        {"chunk_id": 1, "score": 0.8, "content": "..."},
        {"chunk_id": 4, "score": 0.7, "content": "..."},
    ]
    
    rrf_fuser = ReciprocRankFusion()
    final_results = rrf_fuser.fuse(dense_results, bm25_results, top_k=3)
    
    print("RRF融合结果（top-3）:")
    for i, result in enumerate(final_results, 1):
        print(f"{i}. Chunk {result['chunk_id']}, RRF Score: {result['rrf_score']:.4f}")
