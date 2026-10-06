import pandas as pd
from common import relevant, norm_title

files = {"dblp": "dblp_with_abstracts.csv", "s2": "s2.csv", "openalex": "openalex.csv",
         "arxiv": "arxiv.csv", "openreview": "openreview_accepted.csv"}
df = pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in files.values()], ignore_index=True)
print("retrieved:", df.groupby("source").size().to_dict(), "total", len(df))

df = df[(df.title + " " + df.abstract).map(relevant)].copy()
print("after keyword screen:", len(df))

df["tkey"] = df.title.map(norm_title)
# link records that share a DOI or arXiv id to a single title key
for col in ("doi", "arxiv"):
    has = df[df[col] != ""]
    canon = has.groupby(col).tkey.first()
    df.loc[has.index, "tkey"] = has[col].map(canon)

first = lambda s: next((x for x in s if x), "")
merged = df.groupby("tkey").agg(
    title=("title", first), abstract=("abstract", lambda s: max(s, key=len)),
    year=("year", first), venues=("venue", lambda s: " | ".join(sorted({v for v in s if v}))),
    doi=("doi", first), arxiv=("arxiv", first),
    sources=("source", lambda s: ",".join(sorted(set(s)))),
).reset_index(drop=True)
print("unique papers:", len(merged))
merged.to_csv("corpus.csv", index=False)
