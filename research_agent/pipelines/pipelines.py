from research_agent.agents.agents import build_search_agent, build_scrap_agent, writer_chain, critique_chain
from langchain_core.messages import HumanMessage
from rich import print
def build_pipeline(topic):

    state = {}

    # Step 1: Search for relevant information
    print(f"Searching for information on: {topic}")
    search_agent = build_search_agent()
    serach_result =  search_agent.invoke(
        {"messages": HumanMessage(content=f"Search for information on: {topic}")}
    )
    
    state["search_result"] = serach_result["messages"][-1].content
    print(f"Search Result: {state['search_result']}")

    # Step 2: Scrape the URLs from the search results
    print("Scraping URLs from search results...")
    scrap_agent = build_scrap_agent()
    scrap_result = scrap_agent.invoke(
        {"messages": HumanMessage(content=f"Scrape the following URLs: {state['search_result']}")}
    )
    state["scrap_result"] = scrap_result["messages"][-1].content
    print(f"Scrap Result: {state['scrap_result']}") 

    # Step 3: Write a draft based on the scraped information
    print("Writing draft based on scraped information...")
    draft = writer_chain.invoke(
        {"topic": topic, "research": state["scrap_result"]}
    )
    state["draft"] = draft
    print(f"Draft: {state['draft']}")       

    # Step 4: Critique the draft
    print("Critiquing the draft...")
    critique = critique_chain.invoke(
        {"topic": topic, "research": state["scrap_result"], "draft": state["draft"]}
    )   
    state["critique"] = critique
    print(f"Critique: {state['critique']}")
    return state