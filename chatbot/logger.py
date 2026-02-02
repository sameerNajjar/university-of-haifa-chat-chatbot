"""
Simple logging for chatbot interactions
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List

class ChatLogger:
    def __init__(self, log_file: str = "chatbot_log.jsonl"):
        self.log_file = log_file
        
        # Create log file if doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                pass
    
    def log_interaction(
        self,
        query: str,
        answer: str,
        sources: List[tuple],
        query_lang: str,
        response_time: float,
        metadata: Dict[str, Any] = None
    ):
        """Log a single interaction"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'query_lang': query_lang,
            'answer': answer,
            'num_sources': len(sources),
            'avg_source_score': sum(s[0] for s in sources) / len(sources) if sources else 0,
            'top_source_score': sources[0][0] if sources else 0,
            'response_time_sec': round(response_time, 2),
            'answer_length': len(answer),
            'metadata': metadata or {}
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_stats(self, last_n: int = None):
        """Get usage statistics"""
        
        if not os.path.exists(self.log_file):
            return {}
        
        entries = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line))
        
        if last_n:
            entries = entries[-last_n:]
        
        if not entries:
            return {}
        
        stats = {
            'total_queries': len(entries),
            'avg_response_time': sum(e['response_time_sec'] for e in entries) / len(entries),
            'avg_sources_used': sum(e['num_sources'] for e in entries) / len(entries),
            'hebrew_queries': sum(1 for e in entries if e['query_lang'] == 'he'),
            'english_queries': sum(1 for e in entries if e['query_lang'] == 'en'),
        }
        
        return stats