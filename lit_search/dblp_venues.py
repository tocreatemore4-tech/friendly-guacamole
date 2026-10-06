import csv, sys
from collections import Counter
from lxml import etree

VENUE_KEYS = ("conf/uss/", "conf/sp/", "conf/ccs/", "conf/ndss/",
              "conf/nips/", "conf/iclr/", "conf/icml/")
rows, seen = [], Counter()
for _, el in etree.iterparse(sys.argv[1], events=("end",), tag="inproceedings",
                             load_dtd=True, resolve_entities=True, huge_tree=True):
    key, bt, yr = el.get("key"), el.findtext("booktitle") or "", int(el.findtext("year") or 0)
    if 2024 <= yr <= 2026 and (key.startswith(VENUE_KEYS) or "aisec" in bt.lower()):
        ees = [e.text for e in el.findall("ee")]
        doi = next((e.split("doi.org/", 1)[1] for e in ees if "doi.org/" in e), "")
        rows.append({"source": "dblp", "id": key, "venue": bt, "year": yr,
                     "title": "".join(el.find("title").itertext()).rstrip("."),
                     "doi": doi.lower(), "url": ees[0] if ees else ""})
        seen[(key.split("/")[1], bt)] += 1
    el.clear()
    while el.getprevious() is not None:
        del el.getparent()[0]

for (k, bt), n in sorted(seen.items()):
    print(f"{k:8} {bt:40} {n}")
with open("dblp_venues.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(len(rows), "papers")
