"""Small helpers for the Wikimedia Pageviews API and the MediaWiki API.

* ``redirect_titles`` asks a language edition which titles redirect to an
  article, so that views under old or alternative titles can be added back.
  The Pageviews API counts a view under the title that was requested, so an
  article read through a redirect (or under its title before a page move)
  would otherwise be missed.
* ``daily_views`` downloads daily human views (agent=user) for one title and
  caches the raw JSON in data/raw/ (git-ignored).
"""

import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "{project}/all-access/user/{title}/daily/{start}/{end}")
# Wikimedia asks automated clients for a descriptive User-Agent.
HEADERS = {"User-Agent": "did-it-work-synthetic-control/1.0 (portfolio research project)"}


def _get(session, url, params=None):
    for attempt in range(8):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=30)
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(2 ** min(attempt, 5))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = r.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else 2 ** min(attempt, 5))
            continue
        return r
    raise RuntimeError("giving up on %s" % url)


def _cache(name):
    RAW.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return RAW / safe


def redirect_titles(session, lang, title, refresh=False):
    """Return (canonical title, [titles that redirect to it]) on lang.wikipedia."""
    cache = _cache("redirects_%s_%s.json" % (lang, title))
    if cache.exists() and not refresh:
        with open(cache) as f:
            d = json.load(f)
        return d["canonical"], d["redirects"]
    params = dict(action="query", titles=title, redirects=1, prop="redirects",
                  rdlimit="max", rdnamespace=0, format="json")
    r = _get(session, "https://%s.wikipedia.org/w/api.php" % lang, params)
    r.raise_for_status()
    pages = list(r.json()["query"]["pages"].values())
    canonical = pages[0].get("title", title)
    redirects = [x["title"] for x in pages[0].get("redirects", [])]
    with open(cache, "w") as f:
        json.dump(dict(canonical=canonical, redirects=redirects), f, ensure_ascii=False)
    time.sleep(0.3)
    return canonical, redirects


def daily_views(session, lang, title, start, end, refresh=False):
    """Daily views for one title as a Series indexed by date (missing days = 0)."""
    cache = _cache("pv_%s_%s_%s_%s.json" % (lang, title, start, end))
    if cache.exists() and not refresh:
        with open(cache) as f:
            data = json.load(f)
    else:
        url = PAGEVIEWS.format(project="%s.wikipedia.org" % lang,
                               title=quote(title.replace(" ", "_"), safe=""), start=start, end=end)
        r = _get(session, url)
        if r.status_code == 404:          # no views recorded for this title in the window
            data = {"items": []}
        else:
            r.raise_for_status()
            data = r.json()
        with open(cache, "w") as f:
            json.dump(data, f)
        time.sleep(0.3)                   # be polite
    idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    if not data.get("items"):
        return pd.Series(0.0, index=idx)
    s = pd.Series({pd.to_datetime(i["timestamp"][:8]): i["views"] for i in data["items"]}, dtype=float)
    return s.reindex(idx, fill_value=0.0)


def article_views(session, lang, title, start, end, refresh=False):
    """Views of an article summed over its canonical title and every redirect to it."""
    canonical, redirects = redirect_titles(session, lang, title, refresh)
    titles = [canonical] + [t for t in redirects if t != canonical]
    total, by_title = None, {}
    for t in titles:
        s = daily_views(session, lang, t, start, end, refresh)
        by_title[t] = int(s.sum())
        total = s if total is None else total + s
    return total, canonical, by_title
