import re

def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '<img>', text)
    # 1. Remove all property/formatting command tokens
    text = re.sub(r'[\s;]*-?\s*\[[^\]]*\]\([^)]*\);-\S+', '', text)
    
    # 2. Remove portal header (link text after it gets handled in step 4)
    text = text.replace("--------------------- Portal ---------------------", "")
    
    # 3. Remove images entirely
    
    # 4. Links: keep display text, drop URL
    text = re.sub(r'\[([^\]]*)\]\(([^()]*|\([^()]*\))*\)', r'\1', text)

    text = re.sub(r'-\s*\[\s*(?:x|)\s*\]', '-', text)

    # 5. Drop blank/empty bullets (lines that reduce to nothing or just a dash)
    lines = [l for l in text.splitlines() if l.strip().rstrip('-').strip()]
    text = '\n'.join(lines)

    return text
