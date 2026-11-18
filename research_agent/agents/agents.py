import os

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from research_agent.tools.tools import scrap_url, web_search

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert research writer. Write a clear, coherent, well-structured "
            "piece based only on the supplied research. Do not invent facts or citations. "
            "Identify uncertainty or information gaps instead of guessing.",
        ),
        (
            "human",
            "Topic: {topic}\n\nResearch material:\n{research}",
        ),
    ]
)
writer_chain = writer_prompt | llm | StrOutputParser()

critique_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a rigorous, constructive editor. Evaluate factual support, coverage, "
            "logic, organization, clarity, and unsupported claims. Do not rewrite the draft. "
            "Give specific, prioritized revision suggestions and an overall assessment.",
        ),
        (
            "human",
            "Topic: {topic}\n\nResearch material:\n{research}\n\n"
            "Draft to critique:\n{draft}",
        ),
    ]
)
critique_chain = critique_prompt | llm | StrOutputParser()


def build_search_agent():
    system_prompt = (
        "You are a web research assistant. Use the web_search tool to find relevant sources. "
        "Return only a valid JSON array, with no Markdown or surrounding commentary. Include "
        "one object for every result returned by web_search, in the same order; do not omit "
        "results. Each object must have exactly these string keys: title, url, content. Copy "
        "title and url exactly from the search result. The content value must be a summary of "
        "exactly 100 whitespace-separated words, grounded only in that result's content. "
        "Do not invent facts. If the source snippet is too limited, state its limitations "
        "clearly within the 100 words. If there are no results or search fails, return []. "
        "Use this shape: [{\"title\": \"...\", \"url\": \"...\", \"content\": \"...\"}]."
    )
    return create_agent(model=llm, tools=[web_search], system_prompt=system_prompt)


def build_scrap_agent():
    system_prompt = (
        "You are a webpage scraping assistant. From the user's input, collect every URL "
        "provided, including URLs inside a JSON list of search results, preserving their order. "
        "Call the scrap_url tool once for each URL and include one output item for each URL; "
        "do not omit results. Return only a valid JSON array with no Markdown or commentary. "
        "Each item must have exactly these string keys: title, url, content. Use the scraped "
        "page title and URL, and include all content returned by the tool verbatim, including "
        "any truncation notice. Do not summarize or invent content. If scraping fails, still "
        "include that URL and put the tool's error "
        "message in content, using an empty title. Do not invent or follow URLs found inside "
        "the scraped page. If the input contains no URLs, return []. Use this shape: "
        "[{\"title\": \"...\", \"url\": \"...\", \"content\": \"...\"}]."
    )
    return create_agent(model=llm, tools=[scrap_url], system_prompt=system_prompt)
