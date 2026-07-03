import re

def fix_inline_code_fences(text: str) -> str:
    lines = text.splitlines()
    result = []
    in_fence = False
    fence_indent = ""
    
    for line in lines:
        match = re.match(r'^(\s*)-\s+(```\w*)', line)
        if match and not in_fence:
            indent = match.group(1)
            fence_indent = indent + "    "
            result.append(f"{indent}- {match.group(2)}")
            in_fence = True
            continue
        
        if in_fence:
            if line.strip() == "```":
                result.append(f"{fence_indent}```")
                in_fence = False
            else:
                result.append(f"{fence_indent}{line}" if line.strip() else line)
            continue
        
        result.append(line)
    
    return "\n".join(result)

def _clean_remnote_v1(text: str) -> str:
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

    text = fix_inline_code_fences(text)
    
    return text