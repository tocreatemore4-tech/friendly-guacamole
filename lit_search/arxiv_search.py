import csv
from datetime import date, timedelta
import arxiv
from common import AGENT, THREAT, DEFENCE

CATS = "(cat:cs.CR OR cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.MA OR cat:cs.SE)"

def ax_or(terms):
    return "(" + " OR ".join(f'abs:"{t}"' for t in terms) + ")"

def month_windows(start=date(2024, 1, 1), end=date.today()):
    d = start
    while d <= end:
        nxt = date(d.year + d.month // 12, d.month % 12 + 1, 1)
        yield f"{d:%Y%m%d}0000", f"{nxt - timedelta(days=1):%Y%m%d}2359"
        d = nxt

client = arxiv.Client(page_size=500, delay_seconds=3, num_retries=5)
rows = {}
for lo, hi in month_windows():
    q = (f"{ax_or(AGENT)} AND ({ax_or(THREAT)} OR {ax_or(DEFENCE)}) AND {CATS} "
         f"AND submittedDate:[{lo} TO {hi}]")
    if lo.startswith("202401"):
        print(q)
    n = 0
    for r in client.results(arxiv.Search(query=q, max_results=None,
                                         sort_by=arxiv.SortCriterion.SubmittedDate,
                                         sort_order=arxiv.SortOrder.Ascending)):
        aid = r.get_short_id().rsplit("v", 1)[0]
        rows[aid] = {"source": "arxiv", "id": aid, "title": r.title, "abstract": r.summary,
                     "year": r.published.year, "date": r.published.date().isoformat(),
                     "venue": r.journal_ref or "", "doi": (r.doi or "").lower(), "arxiv": aid,
                     "categories": " ".join(r.categories), "url": r.entry_id}
        n += 1
    print(lo[:6], n)

with open("arxiv.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=next(iter(rows.values())).keys())
    w.writeheader(); w.writerows(rows.values())
