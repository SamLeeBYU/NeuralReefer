import re, sys
CMD_RE = re.compile(r'\\rev(add|del)\{')
def strip_revisions(text):
    out = []; i = 0; n = len(text)
    while i < n:
        m = CMD_RE.match(text, i)
        if m:
            kind = m.group(1); j = m.end(); depth = 1; start = j
            while depth > 0:
                if text[j] == '{': depth += 1
                elif text[j] == '}': depth -= 1
                j += 1
            inner = text[start:j-1]
            inner_processed = strip_revisions(inner)
            if kind == 'add': out.append(inner_processed)
            i = j
        else:
            out.append(text[i]); i += 1
    return ''.join(out)
src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding='utf-8') as f: text = f.read()
with open(dst, 'w', encoding='utf-8') as f: f.write(strip_revisions(text))
