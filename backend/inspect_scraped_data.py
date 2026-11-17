
import json
from pathlib import Path

file_path = (
    Path(__file__).parent
    / "scraped_data"
    / "leadonspot_pages.jsonl"
)

with open(file_path, "r", encoding="utf-8") as file:
    pages = [json.loads(line) for line in file if line.strip()]

print(f"\nTotal pages: {len(pages)}")

print("\nALL SCRAPED PAGES")
print("=" * 70)

for index, page in enumerate(pages, 1):
    print(f"\n{index}. {page['title']}")
    print("URL:", page["url"])
    print("Content length:", len(page["content"]))

keywords = [
    "google sheets",
    "decision maker",
    "designation",
    "round robin",
    "inactive leads",
    "stuck deals",
    "data export",
    "lead assignment",
    "workflow automation",
    "follow up",
    "calendar integration",
]

print("\n\nKEYWORD COVERAGE")
print("=" * 70)

for keyword in keywords:
    matches = []

    for page in pages:
        if keyword.lower() in page["content"].lower():
            matches.append(page["title"])

    print(f"\n{keyword}: {len(matches)} matching pages")

    for title in matches[:5]:
        print(" -", title)
