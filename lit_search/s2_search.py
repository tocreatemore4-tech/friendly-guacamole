import csv, os, time, requests
from common import AGENT, THREAT, DEFENCE

S2 = "https://api.semanticscholar.org/graph/v1"
H = {"x-api-key": os.environ["S2_API_KEY"]}
FIELDS = "title,abstract,year,venue,publicationVenue,externalIds,publicationTypes,url"

def s2_or(terms):
    return "(" + " | ".join(f'"{t}"' for t in terms) + ")"

query = f"{s2_or(AGENT)} + ({s2_or(THREAT)} | {s2_or(DEFENCE)})"
params = {"query": query, "year": "2024-2026", "fieldsOfStudy": "Computer Science", "fields": FIELDS}
print(query)

rows, token = [], None
while True:
    if token:
        params["token"] = token
    r = requests.get(f"{S2}/paper/search/bulk", params=params, headers=H, timeout=60)
    if r.status_code == 429:
        time.sleep(10); continue
    r.raise_for_status()
    js = r.json()
    for p in js.get("data", []):
        ext = p.get("externalIds") or {}
        rows.append({"source": "s2", "id": p["paperId"], "title": p["title"],
                     "abstract": p.get("abstract") or "", "year": p.get("year"),
                     "venue": (p.get("publicationVenue") or {}).get("name") or p.get("venue") or "",
                     "doi": (ext.get("DOI") or "").lower(), "arxiv": ext.get("ArXiv") or "",
                     "url": p.get("url")})
    print(f"{len(rows)} / {js.get('total')}")
    token = js.get("token")
    if not token:
        break
    time.sleep(1)

with open("s2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
