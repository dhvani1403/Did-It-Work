"""Download daily Wikipedia views for the Italian ChatGPT ban analysis.

For each outcome (the ChatGPT article and the VPN article) and each language
edition (Italian plus the donor candidates), find the article's title from the
English article's interlanguage links, add up views over the title and every
redirect to it, and save one column per language.

    python scripts/fetch_italy.py            # uses the cache in data/raw/ where possible
    python scripts/fetch_italy.py --refresh  # ignore the cache

Writes data/processed/italy_<outcome>_daily.csv and data/processed/italy_articles.json.
These files are committed, so the analysis runs offline.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import italy_config as cfg  # noqa: E402
import wiki_pageviews as wp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def langlinks(session, en_title, refresh=False):
    cache = wp._cache("langlinks_en_%s.json" % en_title)
    if cache.exists() and not refresh:
        with open(cache) as f:
            return json.load(f)
    params = dict(action="query", titles=en_title, prop="langlinks", lllimit="max", format="json")
    r = wp._get(session, "https://en.wikipedia.org/w/api.php", params)
    r.raise_for_status()
    links = {x["lang"]: x["*"] for p in r.json()["query"]["pages"].values() for x in p.get("langlinks", [])}
    with open(cache, "w") as f:
        json.dump(links, f, ensure_ascii=False)
    return links


def main(refresh=False):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    meta = dict(source=wp.PAGEVIEWS, agent="user", access="all-access",
                start=cfg.START, end=cfg.END, outcomes={})
    with requests.Session() as session:
        for outcome, en_title in cfg.OUTCOMES.items():
            links = langlinks(session, en_title, refresh)
            cols, units = {}, []
            for lang in [cfg.TREATED_LANG] + cfg.DONOR_LANGS:
                if lang not in links:
                    units.append(dict(lang=lang, title=None, note="no article in this language"))
                    print("  %-8s %-3s  no article" % (outcome, lang))
                    continue
                s, canonical, by_title = wp.article_views(session, lang, links[lang], cfg.START, cfg.END, refresh)
                cols[lang] = s
                units.append(dict(lang=lang, title=canonical, n_titles=len(by_title), views_by_title=by_title))
                print("  %-8s %-3s %-32s %3d titles %9d views" % (outcome, lang, canonical[:32], len(by_title), s.sum()))
            df = pd.DataFrame(cols)
            df.index.name = "date"
            df.to_csv(PROCESSED / ("italy_%s_daily.csv" % outcome))
            meta["outcomes"][outcome] = dict(en_title=en_title, units=units)
            print("wrote data/processed/italy_%s_daily.csv  (%d days x %d languages)" % ((outcome,) + df.shape))
    with open(PROCESSED / "italy_articles.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
