"""Throwaway diagnostic: show what triggered each section label.

Not part of the pipeline — a scratchpad for tuning detect_section against
real documents. Delete once #5 closes.
"""

import sys
from collections import Counter

from app.ingestion import _labelled_lines, extract_pages

lines = _labelled_lines(extract_pages(sys.argv[1]))
counts = Counter(line.section for line in lines)

# Only the labels big enough to be suspicious: a real clause is 5-20 lines.
watch = {label for label, n in counts.items() if label and n > 40}

seen: set[str] = set()
for i, line in enumerate(lines):
    if line.section in watch and line.section not in seen:
        seen.add(line.section)
        print(f"=== {line.section}({counts[line.section]} lines, first on p{line.page_number}) ===")
        for follower in lines[i : i + 4]:
            print(f"  {follower.text[:75]!r}")
