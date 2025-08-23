import re
import asyncio
from typing import List

# Unicode ranges for all 22 scheduled languages of India
LANGUAGE_RANGES = {
    'hindi': r'[\u0900-\u097F]',
    'bengali': r'[\u0980-\u09FF]', 
    'telugu': r'[\u0C00-\u0C7F]',
    'marathi': r'[\u0900-\u097F]',
    'tamil': r'[\u0B80-\u0BFF]',
    'urdu': r'[\u0600-\u06FF\u0750-\u077F]',
    'gujarati': r'[\u0A80-\u0AFF]',
    'kannada': r'[\u0C80-\u0CFF]',
    'malayalam': r'[\u0D00-\u0D7F]',
    'odia': r'[\u0B00-\u0B7F]',
    'punjabi': r'[\u0A00-\u0A7F]',
    'assamese': r'[\u0980-\u09FF]',
    'maithili': r'[\u0900-\u097F]',
    'santali': r'[\u1C50-\u1C7F]',
    'kashmiri': r'[\u0600-\u06FF\u0900-\u097F]',
    'nepali': r'[\u0900-\u097F]',
    'konkani': r'[\u0900-\u097F]',
    'manipuri': r'[\uAAE0-\uAAFF]',
    'bodo': r'[\u0900-\u097F]',
    'dogri': r'[\u0900-\u097F]',
    'sanskrit': r'[\u0900-\u097F]',
    'sindhi': r'[\u0600-\u06FF\u0900-\u097F]',
    'english': r'[A-Za-z]'
}

async def detect_language(text: str) -> str:
    language_scores = {}
    
    for language, pattern in LANGUAGE_RANGES.items():
        count = len(re.findall(pattern, text))
        if count > 0:
            language_scores[language] = count
    
    if not language_scores:
        return 'english'
    
    return max(language_scores, key=language_scores.get)

async def normalise_language(question: str) -> str:
    language = await detect_language(question)
    
    if language == 'english':
        question = re.sub(r"[^A-Za-z\s.,!?%()\-–—:;\'\"0-9]", "", question)
    else:
        # Use the detected language's Unicode range
        pattern = LANGUAGE_RANGES[language]
        question = re.sub(f"[^{pattern[1:-1]}\s.,!?%()\-–—:;\'\"0-9]", "", question)

    # Remove newlines and normalize whitespace
    # question = re.sub(r'\n+', ' ', question)
    question = re.sub(r'\s+', ' ', question)
    
    return question.lower().strip()

async def normalise_questions(questions: List[str]) -> List[str]:
    tasks = [normalise_language(q) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)