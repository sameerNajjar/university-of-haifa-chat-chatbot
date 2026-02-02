"""
Language filtering utilities to enforce Hebrew/English only responses
"""
import re

def contains_unwanted_languages(text: str) -> tuple[bool, str]:
    """
    Check if text contains languages other than Hebrew/English
    
    Returns:
        (has_unwanted, details) - tuple of boolean and description
    """
    unwanted_chars = []
    
    # Define unwanted character ranges
    unwanted_ranges = {
        'Arabic': (0x0600, 0x06FF),
        'Russian/Cyrillic': (0x0400, 0x04FF),
        'Chinese': (0x4E00, 0x9FFF),
        'Korean/Hangul': (0xAC00, 0xD7AF),
        'Japanese Hiragana': (0x3040, 0x309F),
        'Japanese Katakana': (0x30A0, 0x30FF),
    }
    
    found_languages = set()
    char_samples = {}
    
    for ch in text:
        code = ord(ch)
        for lang, (start, end) in unwanted_ranges.items():
            if start <= code <= end:
                found_languages.add(lang)
                if lang not in char_samples:
                    char_samples[lang] = []
                if len(char_samples[lang]) < 3:  # Keep 3 sample chars
                    char_samples[lang].append(ch)
    
    if found_languages:
        details = ", ".join([f"{lang} ({' '.join(char_samples[lang])})" 
                            for lang in found_languages])
        return True, details
    
    return False, ""


def clean_response(text: str) -> str:
    """
    Remove unwanted language characters from text
    
    Keeps only:
    - Hebrew (0x0590-0x05FF)
    - English (a-z, A-Z)
    - Numbers (0-9)
    - Common punctuation
    - Whitespace
    """
    allowed_chars = []
    
    for ch in text:
        code = ord(ch)
        
        # Hebrew
        if 0x0590 <= code <= 0x05FF:
            allowed_chars.append(ch)
        # English
        elif ch.isalpha() and ch.isascii():
            allowed_chars.append(ch)
        # Numbers
        elif ch.isdigit():
            allowed_chars.append(ch)
        # Whitespace
        elif ch.isspace():
            allowed_chars.append(ch)
        # Common punctuation
        elif ch in '.,!?;:()\'"[]{}+-=/*@#$%&<>':
            allowed_chars.append(ch)
        # Skip everything else (Arabic, Russian, etc.)
    
    return ''.join(allowed_chars)


def get_language_stats(text: str) -> dict:
    """
    Get statistics about languages in text
    
    Returns:
        dict with counts of each language
    """
    stats = {
        'hebrew': 0,
        'english': 0,
        'arabic': 0,
        'russian': 0,
        'other': 0,
        'total_alpha': 0
    }
    
    for ch in text:
        if not ch.isalpha():
            continue
            
        stats['total_alpha'] += 1
        code = ord(ch)
        
        # Hebrew
        if 0x0590 <= code <= 0x05FF:
            stats['hebrew'] += 1
        # English
        elif ch.isascii():
            stats['english'] += 1
        # Arabic
        elif 0x0600 <= code <= 0x06FF:
            stats['arabic'] += 1
        # Russian/Cyrillic
        elif 0x0400 <= code <= 0x04FF:
            stats['russian'] += 1
        # Other
        else:
            stats['other'] += 1
    
    return stats


def validate_response_language(response: str, user_query: str) -> tuple[bool, str]:
    """
    Validate that response is in Hebrew/English only
    
    Returns:
        (is_valid, error_message)
    """
    has_unwanted, details = contains_unwanted_languages(response)
    
    if has_unwanted:
        return False, f"Response contains unwanted languages: {details}"
    
    stats = get_language_stats(response)
    
    # Check if response has meaningful content
    if stats['total_alpha'] == 0:
        return False, "Response has no alphabetic content"
    
    # Check if unwanted languages make up >5% of content
    unwanted_ratio = (stats['arabic'] + stats['russian'] + stats['other']) / max(1, stats['total_alpha'])
    if unwanted_ratio > 0.05:
        return False, f"Response contains {unwanted_ratio:.1%} unwanted language characters"
    
    return True, ""


# Example usage
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "This is English text",
        "זה טקסט בעברית",
        "This is mixed עברית and English",
        "هذا نص عربي",  # Arabic
        "Это русский текст",  # Russian
        "Mixed: Hebrew עברית, English, and تطبيقات (Arabic)",
    ]
    
    print("Testing language detection:")
    print("=" * 60)
    
    for text in test_texts:
        has_unwanted, details = contains_unwanted_languages(text)
        stats = get_language_stats(text)
        
        print(f"\nText: {text}")
        print(f"Has unwanted: {has_unwanted}")
        if has_unwanted:
            print(f"  Details: {details}")
        print(f"Stats: {stats}")
        
        is_valid, error = validate_response_language(text, "test")
        print(f"Valid: {is_valid}")
        if not is_valid:
            print(f"  Error: {error}")
            print(f"  Cleaned: {clean_response(text)}")