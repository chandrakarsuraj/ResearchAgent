import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from readability import Document

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


def _validate_public_http_url(url: str) -> None:
    """Reject malformed URLs and hosts that resolve to non-public IP addresses."""
    try:
        parsed_url = urlsplit(url)
        port = parsed_url.port
    except ValueError as e:
        raise ValueError("The URL is malformed.") from e

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Only valid HTTP or HTTPS URLs can be scraped.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("URLs containing embedded credentials are not supported.")

    hostname = parsed_url.hostname
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            records = socket.getaddrinfo(
                hostname,
                port or (443 if parsed_url.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
            addresses = {
                ipaddress.ip_address(record[4][0].split("%", 1)[0])
                for record in records
            }
        except (OSError, ValueError) as e:
            raise ValueError("The URL host could not be resolved safely.") from e

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Scraping hosts on private or non-public networks is not allowed.")


@tool
def scrap_url(url: str) -> str:
    """Safely fetch a public webpage and extract bounded readable text."""
    if not isinstance(url, str) or not url.strip():
        return "Please provide a non-empty URL."

    current_url = url.strip()
    response = None
    try:
        redirect_statuses = {301, 302, 303, 307, 308}
        for redirect_count in range(6):
            _validate_public_http_url(current_url)
            response = requests.get(
                current_url,
                headers={"User-Agent": "ResearchAgent/1.0"},
                timeout=(5, 20),
                stream=True,
                allow_redirects=False,
            )
            if response.status_code not in redirect_statuses:
                break

            location = response.headers.get("Location")
            response.close()
            response = None
            if not location:
                return "The webpage returned a redirect without a destination."
            if redirect_count == 5:
                return "The webpage exceeded the maximum number of redirects."
            current_url = urljoin(current_url, location)

        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "html" not in content_type:
            return f"The URL did not return an HTML page (content type: {content_type})."

        max_bytes = 3_000_000
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
        response.close()
        response = None

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

        max_content_chars = 6_000
        if len(content) > max_content_chars:
            content = content[:max_content_chars].rstrip() + "\n[Content truncated at 6000 characters.]"

        title = title or readability_title or "Untitled"
        return f"Title: {title}\nURL: {page_url}\n\n{content}"
    except ValueError as e:
        return f"Cannot scrape URL: {e}"
    except requests.RequestException as e:
        return f"Could not fetch webpage ({type(e).__name__}): {e}"
    except Exception as e:
        return f"An error occurred while scraping the webpage: {e}"
    finally:
        if response is not None:
            response.close()
