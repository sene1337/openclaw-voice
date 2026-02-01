"""
Text utilities for voice-friendly output.

Cleans AI responses for TTS (removes markdown, hashtags, etc.)
while preserving original for display.
"""

import re


def clean_for_speech(text: str) -> str:
    """
    Clean text for TTS rendering.
    
    Removes:
    - Markdown formatting (**, *, #, ```, etc.)
    - Hashtags (#word)
    - URLs
    - Emojis
    - Multiple spaces/newlines
    
    Converts:
    - Bullet points to spoken equivalents
    - Numbers with context
    """
    if not text:
        return text
    
    # Remove code blocks first (``` ... ```)
    text = re.sub(r'```[\s\S]*?```', ' code block omitted ', text)
    
    # Remove inline code (`...`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_
    
    # Remove hashtags (but keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove common emojis (keep some expressive ones?)
    # For now, remove most technical emojis
    text = re.sub(r'[🔗📦📁💻🖥️⚡🔧🛠️📝✅❌⚠️🚀🎯💡🔍📊📈📉🗂️📋]', '', text)
    
    # Convert bullet points to spoken form
    text = re.sub(r'^\s*[-•]\s*', 'Next, ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove numbered lists
    
    # Clean up multiple newlines
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Don't end with "Next, "
    if text.endswith('Next,'):
        text = text[:-5].strip()
    
    return text


def estimate_speech_duration(text: str, wpm: int = 150) -> float:
    """
    Estimate speech duration in seconds.
    
    Args:
        text: Text to speak
        wpm: Words per minute (default 150 for natural speech)
    
    Returns:
        Estimated duration in seconds
    """
    word_count = len(text.split())
    return (word_count / wpm) * 60
