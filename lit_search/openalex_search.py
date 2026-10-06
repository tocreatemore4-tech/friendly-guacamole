import csv, os, time, requests
from common import AGENT, THREAT, DEFENCE

def oa_or(terms):
    return "(" + " OR ".join(f'"{t}"' for t in terms) + ")"

def oa_abstract(inv):
    if not inv:
        return ""
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))

search = f"{oa_or(AGENT)} AND ({oa_or(THREAT)} OR {oa_or(DEFENCE)})"
print(search)
params = {
    "filter": f"title_and_abstract.search:{search},publication_year:2024-2026",
    "select": "id,doi,title,publication_year,type,primary_location,locations,abstract_inverted_index",
    "per-page": 200,
    "cursor": "*",
    "api_key": os.environ["OPENALEX_API_KEY"],
}

rows = []
while params["cursor"]:
    r = requests.get("https://api.openalex.org/works", params=params, timeout=60)
    if r.status_code == 429:
        time.sleep(10); continue
    r.raise_for_status()
    js = r.json()
    for w in js["results"]:
        srcs = [(loc.get("source") or {}).get("display_name") or "" for loc in w.get("locations") or []]
        arxiv = next((loc["landing_page_url"].rsplit("/", 1)[-1] for loc in w.get("locations") or []
                      if "arxiv.org/abs/" in (loc.get("landing_page_url") or "")), "")
        rows.append({"source": "openalex", "id": w["id"], "title": w.get("title") or "",
                     "abstract": oa_abstract(w.get("abstract_inverted_index")),
                     "year": w.get("publication_year"), "type": w.get("type"),
                     "venue": " | ".join(s for s in srcs if s),
                     "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                     "arxiv": arxiv})
    print(f"{len(rows)} / {js['meta']['count']}")
    params["cursor"] = js["meta"].get("next_cursor")

with open("openalex.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
