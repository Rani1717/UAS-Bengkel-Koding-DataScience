import json, sys

with open('notebook/UAS_BENGKOD.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Write to file with UTF-8 BOM to ensure proper encoding
with open('notebook_dump.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            out.write(f"{'='*80}\n")
            out.write(f"=== CELL {i} (code) ===\n")
            out.write(f"{'='*80}\n")
            out.write(src)
            out.write('\n\n')
        elif c['cell_type'] == 'markdown':
            src = ''.join(c['source'])
            out.write(f"{'~'*80}\n")
            out.write(f"~~~ CELL {i} (markdown) ~~~\n")
            out.write(f"{'~'*80}\n")
            out.write(src)
            out.write('\n\n')

print("Done! Written to notebook_dump.txt")
