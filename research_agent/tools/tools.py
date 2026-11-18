import os
import re
from urllib.parse import urlsplit

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from readability import Document
from langchain.tools import tool

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Search the web with Tavily and return formatted results."""
    if not isinstance(query, str) or not query.strip():
        return "Please provide a non-empty search query."

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable: set the TAVILY_API_KEY environment variable."

    try:
        tavily_search = TavilySearch(
            tavily_api_key=api_key,
            max_results=5,
            topic="general",
        )
        response = tavily_search.invoke({"query": query.strip()})
        results = response.get("results", [])

        return "\n---\n".join(
            "Title: {title}\nURL: {url}\nSnippet: {content}".format(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                content=item.get("content", ""),
            )
            for item in results
        )
    except Exception as e:
        return f"An error occurred during the web search: {e}"


@tool
def scrap_url(url: str) -> str:
    """Download a web page and extract its main readable text."""
    if not isinstance(url, str) or not url.strip():
        return "Please provide a non-empty URL."

    url = url.strip()
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        return "Please provide a valid HTTP or HTTPS URL."
    if parsed_url.username or parsed_url.password:
        return "URLs containing embedded credentials are not supported."

    max_bytes = 3_000_000
    try:
        with requests.get(
            url,
            headers={"User-Agent": "ResearchAgent/1.0"},
            timeout=(5, 20),
            stream=True,
        ) as response:
            response.raise_for_status()
            final_url = urlsplit(response.url)
            if final_url.scheme not in {"http", "https"} or not final_url.hostname:
                return "The URL redirected to an invalid destination."

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "html" not in content_type:
                return f"The URL did not return an HTML page (content type: {content_type})."

            html_chunks = []
            downloaded_bytes = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_bytes:
                    return "The webpage is too large to scrape (limit: 3 MB)."
                html_chunks.append(chunk)

            html = b"".join(html_chunks).decode(response.encoding or "utf-8", errors="replace")
            page_url = response.url

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            element.decompose()

        readability_text = ""
        readability_title = ""
        try:
            document = Document(html)
            readability_title = document.short_title()
            readable_soup = BeautifulSoup(document.summary(), "html.parser")
            readability_text = readable_soup.get_text(" ", strip=True)
        except Exception:
            pass

        try:
            extracted_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
            ) or ""
        except Exception:
            extracted_text = ""

        content = extracted_text or readability_text or soup.get_text(" ", strip=True)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if not content:
            return f"No readable page content found at {page_url}."

        title = title or readability_title or "Untitled"
        return f"Title: {title}\nURL: {page_url}\n\n{content}"
    except requests.RequestException as e:
        return f"Could not fetch webpage ({type(e).__name__}): {e}"
    except Exception as e:
        return f"An error occurred while scraping the webpage: {e}"
