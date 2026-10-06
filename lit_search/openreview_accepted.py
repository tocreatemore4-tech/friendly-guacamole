import csv, os
import openreview

client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net",
                                         username=os.environ["OR_USER"],
                                         password=os.environ["OR_PASS"])

VENUES = [
    "ICLR.cc/2024/Conference", "ICLR.cc/2025/Conference", "ICLR.cc/2026/Conference",
    "NeurIPS.cc/2024/Conference", "NeurIPS.cc/2025/Conference", "NeurIPS.cc/2026/Conference",
    "NeurIPS.cc/2024/Datasets_and_Benchmarks_Track", "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track",
    "ICML.cc/2024/Conference", "ICML.cc/2025/Conference", "ICML.cc/2026/Conference",
]

def val(note, field):
    x = note.content.get(field)
    return x.get("value") if isinstance(x, dict) else x

rows = []
for vid in VENUES:
    notes = client.get_all_notes(content={"venueid": vid})   # exact venueid == accepted only
    print(f"{vid:50} {len(notes)} accepted")
    for n in notes:
        kws = val(n, "keywords") or []
        rows.append({"source": "openreview", "id": n.id, "title": val(n, "title") or "",
                     "abstract": (val(n, "abstract") or "") + " " + " ".join(kws),
                     "year": vid.split("/")[1], "venue": val(n, "venue") or vid,
                     "doi": "", "arxiv": "", "url": f"https://openreview.net/forum?id={n.id}"})

with open("openreview_accepted.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
