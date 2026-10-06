import csv, os, time, requests
S2 = "https://api.semanticscholar.org/graph/v1"
H = {"x-api-key": os.environ["S2_API_KEY"]}


rows = list(csv.DictReader(open("dblp_venues.csv")))
abstracts = {}

# 1) Papers with a DOI: batch lookup, 500 at a time
with_doi = [r for r in rows if r["doi"]]
for i in range(0, len(with_doi), 500):
    chunk = with_doi[i:i + 500]
    resp = requests.post(f"{S2}/paper/batch", params={"fields": "title,abstract"},
                         json={"ids": [f"DOI:{r['doi']}" for r in chunk]}, headers=H, timeout=120)
    resp.raise_for_status()
    for r, p in zip(chunk, resp.json()):
        abstracts[r["id"]] = (p or {}).get("abstract") or ""
    time.sleep(1)

# 2) Papers without a DOI (e.g. USENIX Security): title match, one call each
for r in rows:
    if r["doi"]:
        continue
    resp = requests.get(f"{S2}/paper/search/match", params={"query": r["title"], "fields": "title,abstract"},
                        headers=H, timeout=60)
    if resp.status_code == 200:
        abstracts[r["id"]] = resp.json()["data"][0].get("abstract") or ""
    time.sleep(1)

for r in rows:
    r["abstract"] = abstracts.get(r["id"], "")
with open("dblp_with_abstracts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(sum(bool(r["abstract"]) for r in rows), "of", len(rows), "have abstracts")
