#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False

CACHE_DIR = Path("/tmp/ua_search_cache")
CACHE_TTL_SECONDS = 600


def decode_redirect_url(raw_href: str) -> str:
    if not raw_href:
        return ""
    if raw_href.startswith("//"):
        raw_href = "https:" + raw_href
    parsed = urllib.parse.urlparse(raw_href)

    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg:
            return urllib.parse.unquote(uddg[0])

    if "yahoo.com" in parsed.netloc and "/RU=" in raw_href:
        match = re.search(r"/RU=([^/]+)/R[A-Z]=", raw_href)
        if match:
            return urllib.parse.unquote(match.group(1))

    return raw_href


def _is_valid_result(url: str, seen_urls: set) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    if url in seen_urls:
        return False
    blocked_hosts = ("duckduckgo.com", "://yahoo.com", "r.://yahoo.com", "ad_provider=")
    if any(h in url for h in blocked_hosts):
        return False
    return True


def _load_cache(query: str, max_results: int) -> list[dict] | None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{query.lower()}::{max_results}".encode("utf-8")).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age <= CACHE_TTL_SECONDS:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return data
    except Exception:
        pass
    return None


def _save_cache(query: str, max_results: int, results: list[dict]) -> None:
    if not results:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{query.lower()}::{max_results}".encode("utf-8")).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        cache_file.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    req_data = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://tavily.com",
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    seen_urls = set()
    for item in data.get("results", []):
        url = item.get("url", "").strip()
        if not _is_valid_result(url, seen_urls):
            continue
        seen_urls.add(url)
        results.append({
            "title": " ".join(item.get("title", "").split()),
            "url": url,
            "snippet": " ".join(item.get("content", "").split()),
        })
        if len(results) >= max_results:
            break
    return results


def _search_brave(query: str, max_results: int, api_key: str) -> list[dict]:
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://brave.com{encoded_q}&count={max_results}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    seen_urls = set()
    for item in data.get("web", {}).get("results", []):
        res_url = item.get("url", "").strip()
        if not _is_valid_result(res_url, seen_urls):
            continue
        seen_urls.add(res_url)
        results.append({
            "title": " ".join(item.get("title", "").split()),
            "url": res_url,
            "snippet": " ".join(item.get("description", "").split()),
        })
        if len(results) >= max_results:
            break
    return results


def _search_ddgs_lib(query: str, max_results: int) -> list[dict]:
    results = []
    seen_urls = set()

    for backend in ("lite", "auto"):
        try:
            raw_items = list(DDGS().text(query, max_results=max_results + 4, backend=backend))
        except Exception:
            continue

        for item in raw_items:
            url = item.get("href", "").strip()
            if not _is_valid_result(url, seen_urls):
                continue
            seen_urls.add(url)
            results.append({
                "title": " ".join(item.get("title", "").split()),
                "url": url,
                "snippet": " ".join(item.get("body", "").split()),
            })
            if len(results) >= max_results:
                return results

        if results:
            return results

    return results

def _search_ddg_html(query: str, max_results: int) -> list[dict]:
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com{encoded_q}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://duckduckgo.com",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    if "anomaly.js" in html or "challenge-form" in html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen_urls = set()

    for block in soup.select(".result"):
        if "result--ad" in block.get("class", []):
            continue
        title_el = block.select_one(".result__title a, a.result__a")
        snippet_el = block.select_one(".result__snippet")
        if not title_el:
            continue

        real_url = decode_redirect_url(title_el.get("href", "").strip())
        if not _is_valid_result(real_url, seen_urls):
            continue

        seen_urls.add(real_url)
        title = " ".join(title_el.get_text(" ", strip=True).split())
        snippet = " ".join(snippet_el.get_text(" ", strip=True).split()) if snippet_el else ""
        results.append({
            "title": title,
            "url": real_url,
            "snippet": snippet,
        })
        if len(results) >= max_results:
            break

    return results


def _search_yahoo_html(query: str, max_results: int) -> list[dict]:
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://yahoo.com{encoded_q}&ei=UTF-8"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen_urls = set()

    for block in soup.select("div.algo, div.dd.algo"):
        title_el = block.select_one("h3.title a, a")
        snippet_el = block.select_one(".compText, p")
        if not title_el:
            continue

        real_url = decode_redirect_url(title_el.get("href", "").strip())
        if not _is_valid_result(real_url, seen_urls):
            continue

        seen_urls.add(real_url)
        title = " ".join(title_el.get_text(" ", strip=True).split())
        snippet = " ".join(snippet_el.get_text(" ", strip=True).split()) if snippet_el else ""
        results.append({
            "title": title,
            "url": real_url,
            "snippet": snippet,
        })
        if len(results) >= max_results:
            break

    return results
def search_web(query: str, max_results: int = 8) -> list[dict]:
    cached = _load_cache(query, max_results)
    if cached:
        return cached

    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            res = _search_tavily(query, max_results, tavily_key)
            if res:
                _save_cache(query, max_results, res)
                return res
        except Exception:
            pass

    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if brave_key:
        try:
            res = _search_brave(query, max_results, brave_key)
            if res:
                _save_cache(query, max_results, res)
                return res
        except Exception:
            pass

    if HAS_DDGS:
        try:
            res = _search_ddgs_lib(query, max_results)
            if res:
                _save_cache(query, max_results, res)
                return res
        except Exception:
            pass

    for fallback_fn in (_search_ddg_html, _search_yahoo_html):
        try:
            res = fallback_fn(query, max_results)
            if res:
                _save_cache(query, max_results, res)
                return res
        except Exception:
            continue

    return []


def main():
    parser = argparse.ArgumentParser(description="Поиск реальных URL в сети через Python")
    parser.add_argument("query", help="Поисковый запрос в кавычках")
    parser.add_argument("--limit", "-n", type=int, default=8, help="Количество результатов")
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        print("[ERROR] Пустой поисковый запрос.", file=sys.stderr)
        sys.exit(1)

    try:
        items = search_web(query, max_results=args.limit)
    except Exception as e:
        print(f"[ERROR] Ошибка поиска: {e}", file=sys.stderr)
        sys.exit(1)

    if not items:
        print(f"По запросу '{query}' результатов не найдено.")
        sys.exit(0)

    print(f"=== РЕЗУЛЬТАТЫ ПОИСКА ПО ЗАПРОСУ: {query} ===\n")
    for idx, item in enumerate(items, start=1):
        print(f"{idx}. {item['title']}")
        print(f"   URL: {item['url']}")
        if item["snippet"]:
            print(f"   Описание: {item['snippet']}")
        print()


if __name__ == "__main__":
    main()
