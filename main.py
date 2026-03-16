import os
import uuid
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Make sure API key is set
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY environment variable")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# Set up page configuration
st.set_page_config(
    page_title="Gemini Researcher Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📰 Gemini Researcher Agent")
st.subheader("Powered by Google Gemini 2.5 Flash")
st.markdown("""
This app uses Google Gemini to create a multi-step research pipeline that researches
topics and generates comprehensive research reports.
""")

# Define data models
class ResearchPlan(BaseModel):
    topic: str
    search_queries: list[str]
    focus_areas: list[str]

class ResearchReport(BaseModel):
    title: str
    outline: list[str]
    report: str
    sources: list[str]
    word_count: int

GEMINI_MODEL = "gemini-2.5-flash"

def generate(prompt: str, system_instruction: str, use_search: bool = False, json_output: bool = False) -> str:
    """Call Gemini with optional search grounding and/or JSON output."""
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        response_mime_type="application/json" if (json_output and not use_search) else None,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    return response.text

def save_important_fact(fact: str, source: str = None) -> str:
    """Save an important fact discovered during research."""
    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []
    st.session_state.collected_facts.append({
        "fact": fact,
        "source": source or "Not specified",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    return f"Fact saved: {fact}"

def extract_json(text: str) -> dict:
    """Extract JSON from model response, handling markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)

# Create sidebar for input and controls
with st.sidebar:
    st.header("Research Topic")
    user_topic = st.text_input("Enter a topic to research:")

    start_button = st.button("Start Research", type="primary", disabled=not user_topic)

    st.divider()
    st.subheader("Example Topics")
    example_topics = [
        "What are the best cruise lines in USA for first-time travelers who have never been on a cruise?",
        "What are the best affordable espresso machines for someone upgrading from a French press?",
        "What are the best off-the-beaten-path destinations in India for a first-time solo traveler?"
    ]

    for topic in example_topics:
        if st.button(topic, key=topic):
            user_topic = topic
            start_button = True

# Main content area with two tabs
tab1, tab2 = st.tabs(["Research Process", "Report"])

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex[:16]
if "collected_facts" not in st.session_state:
    st.session_state.collected_facts = []
if "research_done" not in st.session_state:
    st.session_state.research_done = False
if "report_result" not in st.session_state:
    st.session_state.report_result = None


def run_research(topic: str):
    """Run the full Triage → Research → Editor pipeline using Gemini."""
    st.session_state.collected_facts = []
    st.session_state.research_done = False
    st.session_state.report_result = None

    with tab1:
        message_container = st.container()

    # ── Step 1: Triage Agent ──────────────────────────────────────────────────
    with message_container:
        st.write("🔍 **Triage Agent**: Planning research approach...")

    triage_prompt = f"""Create a research plan for: {topic}

Return ONLY valid JSON with this exact structure:
{{
  "topic": "clear statement of the research topic",
  "search_queries": ["query1", "query2", "query3", "query4"],
  "focus_areas": ["area1", "area2", "area3", "area4"]
}}"""

    try:
        triage_text = generate(
            prompt=triage_prompt,
            system_instruction=(
                "You are a research coordinator. Given a topic, produce a structured "
                "research plan as a JSON object and nothing else."
            ),
            json_output=True,
        )
        research_plan = ResearchPlan(**extract_json(triage_text))
    except Exception:
        research_plan = ResearchPlan(
            topic=topic,
            search_queries=[
                f"{topic} overview",
                f"{topic} key facts",
                f"{topic} latest developments",
                f"{topic} recommendations"
            ],
            focus_areas=["Overview", "Key findings", "Analysis", "Recommendations"]
        )

    with message_container:
        st.write("📋 **Research Plan**:")
        st.json(research_plan.model_dump())

    # ── Step 2: Research Agent ────────────────────────────────────────────────
    with message_container:
        st.write("🔎 **Research Agent**: Searching the web...")

    all_research = []
    for query in research_plan.search_queries:
        with message_container:
            st.write(f"   Searching: *{query}*")

        summary = generate(
            prompt=query,
            system_instruction=(
                "You are a research assistant. Search the web for the given query and "
                "produce a concise summary: 2-3 paragraphs, under 300 words. "
                "Capture main points only. Do not add commentary beyond the summary."
            ),
            use_search=True,
        )
        all_research.append(f"**Query**: {query}\n\n{summary}")
        save_important_fact(
            fact=summary[:200] + ("..." if len(summary) > 200 else ""),
            source=f"Web search: {query}"
        )

    with message_container:
        st.write("📚 **Collected Research**:")
        for fact in st.session_state.collected_facts:
            st.info(f"**{fact['source']}** — {fact['timestamp']}\n\n{fact['fact']}")

    # ── Step 3: Editor Agent ──────────────────────────────────────────────────
    with message_container:
        st.write("📝 **Editor Agent**: Creating comprehensive research report...")

    research_text = "\n\n---\n\n".join(all_research)
    editor_prompt = f"""Using the research below, write a comprehensive report on: {topic}

RESEARCH:
{research_text}

Return ONLY valid JSON with this exact structure:
{{
  "title": "Report title",
  "outline": ["Section 1 title", "Section 2 title", "Section 3 title"],
  "report": "Full markdown report, at least 1000 words",
  "sources": ["source or URL 1", "source or URL 2"],
  "word_count": 1000
}}"""

    try:
        editor_text = generate(
            prompt=editor_prompt,
            system_instruction=(
                "You are a senior researcher. Write comprehensive, well-structured "
                "research reports in markdown format, at least 1000 words."
            ),
            json_output=True,
        )
        report = ResearchReport(**extract_json(editor_text))
        report.word_count = len(report.report.split())
        st.session_state.report_result = report

        with message_container:
            st.write("✅ **Research Complete! Report Generated.**")
            st.write("📄 **Report Preview**:")
            st.markdown(report.report[:300] + "...")
            st.write("*See the Report tab for the full document.*")

    except Exception as e:
        st.session_state.report_result = editor_text if 'editor_text' in locals() else ""
        with message_container:
            st.write("✅ **Research Complete!** (raw output — structured parsing failed)")
            st.write(f"Parse error: {e}")

    st.session_state.research_done = True


# Run the research when the button is clicked
if start_button and user_topic:
    with st.spinner(f"Researching: {user_topic}"):
        try:
            run_research(user_topic)
        except Exception as e:
            st.error(f"An error occurred during research: {str(e)}")
            st.session_state.report_result = (
                f"# Research on {user_topic}\n\n"
                f"An error occurred during the research process.\n\nError: {str(e)}"
            )
            st.session_state.research_done = True

# Display results in the Report tab
with tab2:
    if st.session_state.research_done and st.session_state.report_result:
        report = st.session_state.report_result

        if isinstance(report, ResearchReport):
            title = report.title

            if report.outline:
                with st.expander("Report Outline", expanded=True):
                    for i, section in enumerate(report.outline):
                        st.markdown(f"{i+1}. {section}")

            st.info(f"Word Count: {report.word_count}")

            report_content = report.report
            st.markdown(report_content)

            if report.sources:
                with st.expander("Sources"):
                    for i, source in enumerate(report.sources):
                        st.markdown(f"{i+1}. {source}")

            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown"
            )
        else:
            report_content = str(report)
            title = user_topic.title() if user_topic else "Research"
            st.title(title)
            st.markdown(report_content)
            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown"
            )
