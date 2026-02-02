"""
Analyze chatbot usage logs
"""
from logger import ChatLogger

logger = ChatLogger("chatbot_interactions.jsonl")

# Get stats for last 100 interactions
stats = logger.get_stats(last_n=100)

print("📊 Chatbot Usage Statistics (Last 100 queries)")
print("=" * 50)
print(f"Total queries: {stats.get('total_queries', 0)}")
print(f"Hebrew queries: {stats.get('hebrew_queries', 0)}")
print(f"English queries: {stats.get('english_queries', 0)}")
print(f"Avg response time: {stats.get('avg_response_time', 0):.2f}s")
print(f"Avg sources used: {stats.get('avg_sources_used', 0):.1f}")