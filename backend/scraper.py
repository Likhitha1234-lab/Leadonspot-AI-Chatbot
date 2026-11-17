import json
import logging
import re
import time
import xml.etree.ElementTree as ET

from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ==================================================
# CONFIGURATION
# ==================================================

BASE_URL = "https://leadonspot.com/"
ALLOWED_DOMAINS = {
    "leadonspot.com",
    "www.leadonspot.com",
}

MAX_PAGES = 300
REQUEST_TIMEOUT = 20
REQUEST_DELAY = 1

MIN_CONTENT_LENGTH = 500

# Pages built with JavaScript tabs/animations: always render in a browser.
BROWSER_FIRST_PATHS = ("/products/", "/solutions", "/contact", "/demo", "/faqs", "/pricing")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "scraped_data"

OUTPUT_FILE = OUTPUT_DIR / "leadonspot_pages.jsonl"
REPORT_FILE = OUTPUT_DIR / "scraping_report.json"

USER_AGENT = (
    "LeadonspotKnowledgeBot/1.0 "
    "(public website content crawler)"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

session = requests.Session()
session.headers.update(HEADERS)


# ==================================================
# ROBOTS.TXT
# ==================================================

def load_robots():
    robots_url = urljoin(BASE_URL, "/robots.txt")

    parser = RobotFileParser()

    try:
        response = session.get(
            robots_url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        parser.parse(response.text.splitlines())

        logging.info("Successfully loaded robots.txt")

        return parser

    except requests.RequestException as error:
        raise RuntimeError(
            "Could not verify website crawling permissions "
            f"from robots.txt: {error}"
        )


# ==================================================
# URL NORMALIZATION
# ==================================================

def normalize_url(url):
    if not url:
        return None

    url, _ = urldefrag(url)

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    hostname = (parsed.hostname or "").lower()

    if hostname not in ALLOWED_DOMAINS:
        return None

    path = re.sub(r"/+", "/", parsed.path or "/")

    # Use one canonical hostname.
    return urlunparse((
        "https",
        "leadonspot.com",
        path,
        "",
        "",
        "",
    ))


def is_valid_page(url):
    parsed = urlparse(url)

    excluded_extensions = (
        ".pdf", ".jpg", ".jpeg", ".png",
        ".gif", ".svg", ".webp", ".css",
        ".js", ".zip", ".xml", ".mp4",
        ".woff", ".woff2", ".ttf",
    )

    if parsed.path.lower().endswith(excluded_extensions):
        return False

    excluded_paths = (
        "/login",
        "/signin",
        "/sign-in",
        "/signup",
        "/register",
        "/checkout",
        "/cart",
        "/account",
        "/admin",
    )

    path = parsed.path.lower()

    if any(path.startswith(item) for item in excluded_paths):
        return False

    return True


# ==================================================
# SITEMAP DISCOVERY
# ==================================================

def discover_sitemap_urls():
    sitemap_locations = [
        urljoin(BASE_URL, "/sitemap.xml"),
        urljoin(BASE_URL, "/sitemap_index.xml"),
    ]

    discovered_urls = set()
    visited_sitemaps = set()

    while sitemap_locations:
        sitemap_url = sitemap_locations.pop(0)

        if sitemap_url in visited_sitemaps:
            continue

        visited_sitemaps.add(sitemap_url)

        try:
            response = session.get(
                sitemap_url,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            root = ET.fromstring(response.content)

            for element in root.iter():
                if not element.tag.endswith("loc"):
                    continue

                location = (element.text or "").strip()

                if not location:
                    continue

                if location.lower().endswith(".xml"):
                    sitemap_locations.append(location)
                    continue

                normalized = normalize_url(location)

                if normalized and is_valid_page(normalized):
                    discovered_urls.add(normalized)

        except (
            requests.RequestException,
            ET.ParseError,
        ) as error:
            logging.warning(
                "Could not read sitemap %s: %s",
                sitemap_url,
                error,
            )

    logging.info(
        "Sitemap URLs discovered: %s",
        len(discovered_urls),
    )

    return discovered_urls


# ==================================================
# HTML CONTENT EXTRACTION
# ==================================================

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "header",
        "footer",
        "nav",
        "form",
    ]):
        element.decompose()

    for element in soup.select(
        ".navbar, .footer, .cookie-banner, "
        ".popup, .modal, .advertisement, "
        "[aria-hidden='true']"
    ):
        element.decompose()

    return soup


def extract_page_content(html, url):
    soup = clean_html(html)

    title = ""

    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    # Prefer the main page content when available.
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # Keep useful structure such as headings, lists and tables.
    structured_parts = []

    seen_text = set()
    block_tags = ["p", "li", "div", "h1", "h2", "h3", "h4", "table", "ul"]

    for element in main_content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th",
         "dt", "dd", "summary", "button", "a", "span", "div"],
        recursive=True,
    ):
        # Only leaf-like elements, otherwise text repeats at every level.
        if element.name in ("span", "div", "a", "button", "li") and element.find(block_tags):
            continue

        text = element.get_text(" ", strip=True)

        if text and text not in seen_text:
            seen_text.add(text)
            structured_parts.append(text)

    structured_text = "\n".join(structured_parts)

    # Trafilatura provides another extraction path.
    extracted_text = trafilatura.extract(
        str(soup),
        url=url,
        include_links=False,
        include_tables=True,
        include_comments=False,
        favor_precision=False,
    ) or ""

    # Choose the longer useful extraction.
    content = (
        structured_text
        if len(structured_text) > len(extracted_text)
        else extracted_text
    )

    if not content:
        content = main_content.get_text(
            separator="\n",
            strip=True,
        )

    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    return title, content.strip()


# ==================================================
# INTERNAL LINK EXTRACTION
# ==================================================

def extract_internal_links(html, current_url):
    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for anchor in soup.find_all("a", href=True):
        absolute_url = urljoin(
            current_url,
            str(anchor["href"]),
        )

        normalized = normalize_url(absolute_url)

        if normalized and is_valid_page(normalized):
            links.add(normalized)

    return links


# ==================================================
# PLAYWRIGHT FALLBACK
# ==================================================

class BrowserExtractor:

    def __init__(self):
        self.playwright = None
        self.browser = None

    def start(self):
        self.playwright = sync_playwright().start()

        self.browser = self.playwright.chromium.launch(
            headless=True
        )

    def extract(self, url):
        if self.browser is None:
            return None

        page = self.browser.new_page(
            user_agent=USER_AGENT
        )

        try:
            response = page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=8000,
                )
            except PlaywrightTimeoutError:
                pass

            # Scroll to trigger lazy sections, open tabs / accordions.
            for _ in range(6):
                page.mouse.wheel(0, 2500)
                page.wait_for_timeout(400)

            for selector in ["[role=tab]", "summary",
                             "button[aria-expanded=false]"]:
                for element in page.query_selector_all(selector)[:40]:
                    try:
                        element.click(timeout=800)
                        page.wait_for_timeout(150)
                    except Exception:
                        pass

            html = page.content()

            return html

        except Exception as error:
            logging.warning(
                "Playwright failed for %s: %s",
                url,
                error,
            )

            return None

        finally:
            page.close()

    def close(self):
        if self.browser:
            self.browser.close()

        if self.playwright:
            self.playwright.stop()


# ==================================================
# PAGE SCRAPER
# ==================================================

def scrape_page(url, robots, browser):
    if not robots.can_fetch(USER_AGENT, url):
        logging.info("Blocked by robots.txt: %s", url)
        return None, set(), "blocked"

    html = None
    extraction_method = "requests"

    # First attempt: Requests.
    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "text/html" not in content_type:
            return None, set(), "not_html"

        html = response.text

    except requests.RequestException as error:
        logging.warning(
            "Requests failed for %s: %s",
            url,
            error,
        )

    title = ""
    content = ""

    if html:
        title, content = extract_page_content(
            html,
            url,
        )

    # Use Playwright when the initial extraction is too small.
    needs_browser = any(
        part in url for part in BROWSER_FIRST_PATHS
    ) and len(content) < 6000

    if not html or len(content) < MIN_CONTENT_LENGTH or needs_browser:
        logging.info(
            "Using Playwright fallback: %s",
            url,
        )

        browser_html = browser.extract(url)

        if browser_html:
            browser_title, browser_content = extract_page_content(
                browser_html,
                url,
            )

            if len(browser_content) > len(content):
                html = browser_html
                title = browser_title
                content = browser_content
                extraction_method = "playwright"

    if not html:
        return None, set(), "failed"

    links = extract_internal_links(
        html,
        url,
    )

    if len(content) < 150:
        return None, links, "insufficient_content"

    page_data = {
        "url": url,
        "title": title or url,
        "content": content,
        "extraction_method": extraction_method,
    }

    return page_data, links, extraction_method


# ==================================================
# MAIN CRAWLER
# ==================================================

def scrape_website():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    robots = load_robots()

    sitemap_urls = discover_sitemap_urls()

    start_url = normalize_url(BASE_URL)

    queue = deque([start_url])

    queue.extend(sorted(sitemap_urls))

    visited = set()
    saved_pages = []

    report = {
        "pages_visited": 0,
        "pages_extracted": 0,
        "requests_extractions": 0,
        "playwright_extractions": 0,
        "insufficient_content": [],
        "failed_pages": [],
        "blocked_pages": [],
        "pages": [],
    }

    browser = BrowserExtractor()

    try:
        browser.start()

        while queue and len(visited) < MAX_PAGES:
            url = queue.popleft()

            if not url or url in visited:
                continue

            visited.add(url)

            logging.info(
                "Crawling %s (%s/%s)",
                url,
                len(visited),
                MAX_PAGES,
            )

            page, links, status = scrape_page(
                url,
                robots,
                browser,
            )

            report["pages_visited"] += 1

            for link in links:
                if link not in visited:
                    queue.append(link)

            if page:
                saved_pages.append(page)

                report["pages_extracted"] += 1

                if page["extraction_method"] == "playwright":
                    report["playwright_extractions"] += 1
                else:
                    report["requests_extractions"] += 1

                report["pages"].append({
                    "url": page["url"],
                    "title": page["title"],
                    "content_length": len(page["content"]),
                    "extraction_method": page["extraction_method"],
                })

                logging.info(
                    "Extracted %s (%s characters)",
                    page["title"],
                    len(page["content"]),
                )

            elif status == "blocked":
                report["blocked_pages"].append(url)

            elif status == "insufficient_content":
                report["insufficient_content"].append(url)

            else:
                report["failed_pages"].append(url)

            time.sleep(REQUEST_DELAY)

    finally:
        browser.close()

    # Save extracted pages.
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for page in saved_pages:
            file.write(
                json.dumps(
                    page,
                    ensure_ascii=False,
                ) + "\n"
            )

    # Save extraction report.
    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    logging.info("=" * 60)
    logging.info("WEBSITE SCRAPING COMPLETED")
    logging.info("Pages visited: %s", report["pages_visited"])
    logging.info("Pages extracted: %s", report["pages_extracted"])
    logging.info(
        "Requests extractions: %s",
        report["requests_extractions"],
    )
    logging.info(
        "Playwright extractions: %s",
        report["playwright_extractions"],
    )
    logging.info(
        "Insufficient content: %s",
        len(report["insufficient_content"]),
    )
    logging.info(
        "Failed pages: %s",
        len(report["failed_pages"]),
    )
    logging.info("Output: %s", OUTPUT_FILE)
    logging.info("Report: %s", REPORT_FILE)
    logging.info("=" * 60)

    return saved_pages


if __name__ == "__main__":
    scrape_website()