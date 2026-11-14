#!/usr/bin/env python3
import argparse
import re
import sys
import urllib.request
import urllib.parse
from html.parser import HTMLParser


def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read()
        return raw.decode(charset, errors="replace")


def parse_with_bs4(html: str, base_url: str, query: str = "", limit_chars: int = 6000) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "Без заголовка"
    lines = []
    seen = set()

    for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "article", "span", "div"]):
        if elem.name in ("div", "span") and elem.find(["p", "li", "h1", "h2", "h3", "article"]):
            continue
        text = " ".join(elem.get_text(" ", strip=True).split())
        if len(text) < 4 or text in seen:
            continue
        if query and query.lower() not in text.lower():
            continue
        seen.add(text)
        lines.append(text)

    links = []
    seen_links = set()
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base_url, a["href"].strip())
        link_text = " ".join(a.get_text(" ", strip=True).split())
        if not href.startswith(("http://", "https://")) or href in seen_links:
            continue
        if query and query.lower() not in (link_text + " " + href).lower():
            continue
        seen_links.add(href)
        links.append(f"- {link_text[:80] or 'Ссылка'} -> {href}")

    output = [f"URL: {base_url}", f"TITLE: {title}", "\n=== ТЕКСТОВЫЕ ДАННЫЕ ==="]
    output.extend(lines[:120] if lines else ["(Текстовых совпадений не найдено)"])

    if links:
        output.append("\n=== РЕЛЕВАНТНЫЕ ССЫЛКИ ===")
        output.extend(links[:40])

    full_text = "\n".join(output)
    return full_text[:limit_chars]


class FallbackTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            cleaned = " ".join(data.split())
            if len(cleaned) >= 4:
                self.chunks.append(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Универсальный парсер веб-страниц (BeautifulSoup)")
    parser.add_argument("url", help="Полный URL страницы (http:// или https://)")
    parser.add_argument("--query", "-q", default="", help="Фильтр по ключевому слову")
    parser.add_argument("--limit", "-l", type=int, default=6000, help="Лимит символов вывода")
    args = parser.parse_args()

    url = args.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить {url}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = parse_with_bs4(html, url, query=args.query, limit_chars=args.limit)
        print(result)
    except ImportError:
        extractor = FallbackTextExtractor()
        extractor.feed(html)
        text = "\n".join(dict.fromkeys(extractor.chunks))
        print(f"URL: {url}\n=== ТЕКСТ (fallback) ===\n{text[:args.limit]}")


if __name__ == "__main__":
    main()
