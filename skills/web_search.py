#!/usr/bin/env python3
import argparse
import sys
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


def decode_ddg_url(raw_href: str) -> str:
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
    return raw_href


def search_web(query: str, max_results: int = 8) -> list[dict]:
    encoded_q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com{encoded_q}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen_urls = set()

    for block in soup.select(".result"):
        title_el = block.select_one(".result__title a, a.result__a")
        snippet_el = block.select_one(".result__snippet")
        if not title_el:
            continue

        real_url = decode_ddg_url(title_el.get("href", "").strip())
        if not real_url.startswith(("http://", "https://")) or real_url in seen_urls:
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
