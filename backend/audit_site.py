"""Shows which scraped pages are thin (probably JS-hidden content).
Run:  python audit_site.py
Then re-run scraper.py and ingest.py."""
import json
from pathlib import Path

rows = [json.loads(l) for l in open(
    Path(__file__).parent / "scraped_data" / "leadonspot_pages.jsonl",
    encoding="utf-8") if l.strip()]

print(f"{'chars':>6}  method      url")
for r in sorted(rows, key=lambda r: len(r["content"])):
    flag = "  <-- THIN" if len(r["content"]) < 1500 else ""
    print(f"{len(r['content']):>6}  {r['extraction_method']:<10}  {r['url']}{flag}")