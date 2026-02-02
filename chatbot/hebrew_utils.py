"""
Hebrew-specific utilities for better search
"""
from typing import List
import re

def expand_hebrew_query(query: str) -> List[str]:
    """
    Generate morphological variations of Hebrew query
    
    Examples:
        "שכר לימוד" → ["שכר לימוד", "שכרי לימוד", ...]
        "במחשב" → ["במחשב", "מחשב"]
    """
    queries = [query.strip()]
    words = query.split()
    
    expanded_words = []
    for word in words:
        variants = [word]
        
        # Remove common prefixes (ב, ל, מ, ה, ש, כ, ו)
        prefixes = ['ב', 'ל', 'מ', 'ה', 'ש', 'כ', 'ו']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > 2:
                variants.append(word[1:])  # without prefix
                break
        
        # Handle plural/singular (basic patterns)
        if word.endswith('ים') and len(word) > 3:  # masculine plural
            variants.append(word[:-2])
        elif word.endswith('ות') and len(word) > 3:  # feminine plural
            variants.append(word[:-2])
        
        expanded_words.append(variants)
    
    # Generate combinations (just use first variant of each word)
    # You can make this more sophisticated
    if len(expanded_words) <= 3:
        # For short queries, try different combinations
        for i, variants in enumerate(expanded_words):
            if len(variants) > 1:
                new_words = words.copy()
                new_words[i] = variants[1]
                queries.append(' '.join(new_words))
    
    return list(set(queries))  # remove duplicates


def detect_query_intent(query: str) -> str:
    """
    Detect what kind of question this is
    
    Returns:
        'factual' | 'procedural' | 'numerical' | 'general'
    """
    query_lower = query.lower()
    
    # Numerical questions (prices, dates, etc.)
    numerical_keywords = [
        'כמה', 'מחיר', 'עלות', 'שכר', 'תאריך', 'מועד',
        'how much', 'price', 'cost', 'when', 'date'
    ]
    if any(kw in query_lower for kw in numerical_keywords):
        return 'numerical'
    
    # Procedural (how-to)
    procedural_keywords = [
        'איך', 'כיצד', 'תהליך', 'שלבים', 'מה צריך',
        'how', 'process', 'steps', 'procedure'
    ]
    if any(kw in query_lower for kw in procedural_keywords):
        return 'procedural'
    
    # Factual (what/who/where)
    factual_keywords = [
        'מה זה', 'מי', 'איפה', 'מהו',
        'what is', 'who', 'where', 'which'
    ]
    if any(kw in query_lower for kw in factual_keywords):
        return 'factual'
    
    return 'general'