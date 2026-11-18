from collections.abc import Callable

from langchain_core.messages import HumanMessage
from rich import print

from research_agent.agents.agents import (
    build_scrap_agent,
    build_search_agent,
    critique_chain,
    writer_chain,
)


def build_pipeline(
    topic: str,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, str]:
    """Run the research pipeline and emit each completed state if requested."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("A non-empty research topic is required.")
    topic = topic.strip()

    def report(stage: str, value: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, value)

    state = {}

    # Step 1: Search for relevant information
    print(f"Searching for information on: {topic}")
    report("status", "Searching the web...")
    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {"messages": [HumanMessage(content=f"Search for information on: {topic}")]}
    )

    state["search_result"] = search_result["messages"][-1].content
    print(f"Search Result: {state['search_result']}")
    report("search_result", state["search_result"])

    # Step 2: Scrape the URLs from the search results
    print("Scraping URLs from search results...")
    report("status", "Scraping the search results...")
    scrap_agent = build_scrap_agent()
    scrap_result = scrap_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Scrape the following URLs: {state['search_result']}"
                )
            ]
        }
    )
    state["scrap_result"] = scrap_result["messages"][-1].content
    print(f"Scrap Result: {state['scrap_result']}")
    report("scrap_result", state["scrap_result"])

    # Step 3: Write a draft based on the scraped information
    print("Writing draft based on scraped information...")
    report("status", "Writing the draft...")
    draft = writer_chain.invoke(
        {"topic": topic, "research": state["scrap_result"]}
    )
    state["draft"] = draft
    report("draft", draft)
    print(f"Draft: {state['draft']}")       

    # Step 4: Critique the draft
    print("Critiquing the draft...")
    report("status", "Critiquing the draft...")
    critique = critique_chain.invoke(
        {"topic": topic, "research": state["scrap_result"], "draft": state["draft"]}
    )   
    state["critique"] = critique
    report("critique", critique)
    print(f"Critique: {state['critique']}")
    return state