# Literature search pipeline (2024–2026)

Scripts that run the survey's keyword search against DBLP, Semantic Scholar,
OpenAlex, arXiv and OpenReview (accepted papers only), then merge everything
into one deduplicated corpus.

```
pip install -r requirements.txt
```

## 0. Keywords (`common.py`)

Edit the three term lists: `AGENT` (agentic components), `THREAT` (terminology
and attack vectors) and `DEFENCE` (defence-centric terms). Every source is
queried with the same logic:

    AGENT  AND  (THREAT  OR  DEFENCE)

`relevant()` applies that rule again locally to title + abstract, so each
source ends up with one identical inclusion criterion.

## 1. DBLP — peer-reviewed venues

1. Download the dump and its DTD into the same folder (pin a monthly release
   from https://dblp.org/xml/release/ and cite it in the paper):
   ```
   wget https://dblp.org/xml/dblp.xml.gz https://dblp.org/xml/dblp.dtd
   gunzip dblp.xml.gz
   ```
2. `python dblp_venues.py dblp.xml` keeps every 2024–2026 paper from
   USENIX Security, S&P, CCS, NDSS, NeurIPS, ICLR, ICML and AISec →
   `dblp_venues.csv`. It prints a count per (dblp key, booktitle). Check it:
   `conf/ccs` also holds other CCS workshops, so drop booktitles you don't want.
3. DBLP has no abstracts. `S2_API_KEY=... python dblp_abstracts.py` fetches
   them from Semantic Scholar (batch lookup by DOI, title match for papers
   without one, e.g. USENIX Security) → `dblp_with_abstracts.csv`.

## 2. Semantic Scholar Graph API

1. Request a free key: https://www.semanticscholar.org/product/api
2. `S2_API_KEY=... python s2_search.py` → `s2.csv`. It uses `/paper/search/bulk`
   (boolean: `|` OR, `+` AND, quoted phrases, matches title + abstract), with
   `year=2024-2026` and a continuation token for paging.
3. Venue names are inconsistent in S2, so don't use the `venue=` filter.
   Inspect `s2.csv`'s `venue` column afterwards.

## 3. OpenAlex

1. Get an API key (required since Feb 2026; a free key has a daily usage
   budget, and search calls cost more than plain filters).
2. `OPENALEX_API_KEY=... python openalex_search.py` → `openalex.csv`. It uses
   the `title_and_abstract.search` filter (boolean `AND`/`OR`, uppercase) plus
   `publication_year:2024-2026`, with cursor paging at 200 results per page.
   Abstracts are rebuilt from `abstract_inverted_index`.

## 4. arXiv (preprints)

`python arxiv_search.py` → `arxiv.csv`.

- Uses `abs:"term"` fields, restricted to `cs.CR/AI/LG/CL/MA/SE`, and
  `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` (first-version date).
- It runs one query per month. That keeps each query under the API's ~30k
  result cap, and a failure only costs one month.
- It waits 3 s between requests, as arXiv asks. Don't parallelise.

## 5. OpenReview — accepted papers only

1. Make an OpenReview account (guest access is throttled).
2. `OR_USER=... OR_PASS=... python openreview_accepted.py` → `openreview_accepted.csv`.
   - It filters on `content.venueid == "<Venue>/<Year>/Conference"`. Accepted
     papers carry exactly that id, while rejected, withdrawn and desk-rejected
     papers get `.../Rejected_Submission`, `.../Withdrawn_Submission`, and so on.
   - Each venue's count is printed. Compare it with the official acceptance
     numbers. A venue id that returns 0 is wrong or not yet public: open
     `https://openreview.net/group?id=<venue id>` to check it.
   - It downloads all accepted papers and keyword-filters them locally in
     step 6, because OpenReview has no boolean search.

## 6. Merge, screen, dedupe

`python merge.py` → `corpus.csv`.

- It applies `relevant()` to every record and prints counts before and after.
  Copy those numbers into a PRISMA flow diagram.
- It merges duplicates by DOI, arXiv id and normalised title, so a preprint and
  its published version become one row. The `sources` and `venues` columns
  show where each paper was found.
- Then do manual or LLM-assisted screening into
  surveys/SoKs/benchmarks, attacks and defences.

Record each search's date, the exact query strings (printed by the scripts)
and the DBLP release for the methodology section.
