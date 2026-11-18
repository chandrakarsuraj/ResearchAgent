import streamlit as st

from research_agent.pipelines.pipelines import build_pipeline


STAGES = {
    "search_result": "Web search results",
    "scrap_result": "Scraped pages",
    "draft": "Writer draft",
    "critique": "Critique",
}
STAGE_ORDER = tuple(STAGES)

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")
st.title("Research Agent")
st.write("Enter a topic to search the web, scrape sources, write a draft, and review it.")

with st.form("research_topic_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="For example: How are Indian students performing in AI and ML?",
    )
    submitted = st.form_submit_button("Start research")

if submitted:
    if not topic.strip():
        st.warning("Enter a research topic to continue.")
    else:
        st.session_state["pipeline_state"] = {}
        st.session_state["pipeline_topic"] = topic.strip()

        progress = st.progress(0.0)
        status = st.empty()
        stage_slots = {stage: st.empty() for stage in STAGE_ORDER}
        completed = [0]

        def show_progress(stage: str, value: str) -> None:
            if stage == "status":
                status.info(value)
                return
            if stage not in STAGES:
                return

            current_state = dict(st.session_state.get("pipeline_state", {}))
            current_state[stage] = value
            st.session_state["pipeline_state"] = current_state

            with stage_slots[stage].container():
                with st.expander(STAGES[stage], expanded=True):
                    if stage in {"search_result", "scrap_result"}:
                        st.code(value, language="json")
                    else:
                        st.markdown(value)

            completed[0] += 1
            progress.progress(completed[0] / len(STAGE_ORDER))

        try:
            with st.spinner("Running the research pipeline..."):
                result = build_pipeline(topic.strip(), progress_callback=show_progress)
            st.session_state["pipeline_state"] = result
            status.success("Research pipeline complete.")
        except Exception as exc:
            status.error("The research pipeline failed.")
            st.error(str(exc))

elif st.session_state.get("pipeline_state"):
    st.subheader(f"Latest results: {st.session_state.get('pipeline_topic', '')}")
    for stage in STAGE_ORDER:
        value = st.session_state["pipeline_state"].get(stage)
        if value is None:
            continue
        with st.expander(STAGES[stage], expanded=(stage == "draft")):
            if stage in {"search_result", "scrap_result"}:
                st.code(value, language="json")
            else:
                st.markdown(value)
