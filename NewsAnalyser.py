"""AI News Analyser: fetches headlines, asks a local LLM (Ollama) to analyse them,
and shows the result in a Streamlit page.

Layout of this file (top to bottom):
    1. Constants
    2. Settings (load / save)
    3. Data: fetching headlines, past analysis, saving results
    4. LLM: prompt building, calling the model, parsing its reply
    5. UI: sidebar, results rendering, page setup
    6. Orchestration and entry point
"""

import json
import os
import pandas as pd
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

import ollama
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# 1. Constants
# --------------------------------------------------------------------------- #

SETTINGS_FILE: str = "settings.json"
TIMESTAMP_FORMAT: str = "%d-%m-%Y"
DEFAULT_FEED_URL: str = "http://feeds.bbci.co.uk/news/world/rss.xml"
OLLAMA_URL: str = "http://localhost:11434"
MAX_PAST_ANALYSIS_AGE_DAYS: int = 7
SECONDS_IN_DAY: int = 86400
SECONDS_IN_HOUR: int = 3600
SECONDS_IN_MINUTE: int = 60

POSTURE_COLOURS: dict[str, str] = {"BUY": "green", "SELL": "red", "WATCH": "orange", "AVOID": "gray"}

PAGE_CSS: str = """<style> [data-testid="stMainBlockContainer"] { max-width: 50vw; } </style>"""


# --------------------------------------------------------------------------- #
# 2. Settings
# --------------------------------------------------------------------------- #

@dataclass
class Settings:
    feed_url: str = DEFAULT_FEED_URL
    timeout: int = 10
    global_themes_count: int = 2
    affected_sectors_count: int = 3
    affected_companies_count: int = 5
    model: str = "llama3.1"
    use_previous_analysis: bool = True
    auto_run: bool = False
    interval: str = "00:15:00"
    save_analysis: bool = True
    output_file: str = "market_opportunities"
    json_indent: int = 4

    @classmethod
    def load(cls, path: str = SETTINGS_FILE) -> "Settings":
        """Load saved settings, falling back to defaults for anything missing."""
        try:
            with open(path, "r") as f:
                saved = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            saved = {}

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in saved.items() if k in known})

    def save(self, path: str = SETTINGS_FILE) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)


# --------------------------------------------------------------------------- #
# 3. Data
# --------------------------------------------------------------------------- #

def fetch_headlines(url: str, timeout: int) -> list[str]:
    """Return all item titles from an RSS feed. Raises on failure."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    titles = (item.findtext("title") for item in root.iter("item"))
    return [title.strip() for title in titles if title]

def load_past_analysis(output_file: str, use_previous_analysis: bool) -> dict:
    """Return the previously saved analysis, or {} if there isn't a usable one."""
    if not use_previous_analysis:
        return {}

    output_file = output_file if output_file.lower().endswith(".json") else f"{output_file}.json"
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def past_analysis_age_days(analysis: dict) -> float | None:
    """Age of a saved analysis in days, or None if it has no valid timestamp."""
    try:
        saved_at = datetime.strptime(analysis["timestamp"], TIMESTAMP_FORMAT)
    except (KeyError, ValueError, TypeError):
        return None

    return (datetime.now() - saved_at).total_seconds() / SECONDS_IN_DAY

def find_new_headlines(headlines: list[str], past_analysis: dict) -> list[str]:
    """Headlines that were not part of the past analysis (all of them if there is none)."""
    past_lines = {line.strip() for line in str(past_analysis.get("processed_headlines", "")).splitlines()}
    return [headline for headline in headlines if f"- {headline}" not in past_lines]

def get_total_seconds_from_time(time_obj: datetime.time) -> int:
    """Convert a datetime.time object to total seconds."""
    return time_obj.hour * SECONDS_IN_HOUR + time_obj.minute * SECONDS_IN_MINUTE + time_obj.second

def has_updates(results: dict) -> bool:
    text = str(results.get("updates_from_past_analysis", "")).strip().strip(".'\"")
    return text.lower() not in {"", "none", "n/a", "null"}

def is_ollama_running() -> bool:
    try:
        return requests.get(OLLAMA_URL, timeout=2).ok
    except requests.RequestException:
        return False

def save_analysis_to_disk(results: dict, output_file: str, indent: int) -> str:
    """Write the analysis to JSON and return the absolute path."""
    output_file = output_file if output_file.lower().endswith(".json") else f"{output_file}.json"
    path = os.path.abspath(output_file)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=indent)
    return path


# --------------------------------------------------------------------------- #
# 4. LLM
# --------------------------------------------------------------------------- #

def build_system_prompt(
    past_analysis: dict,
    age_days: float | None,
    headlines_block: str,
    new_headlines_block: str,
    themes_count: int,
    sectors_count: int,
    companies_count: int,
) -> str:
    age_text = "not found" if age_days is None else f"is {age_days:.1f} day/s old"

    past_for_prompt = {
        k: v for k, v in past_analysis.items()
        if k not in ("timestamp", "processed_headlines")
    }

    return f"""
    Analyse the supplied news headlines and identify genuine investment-relevant economic impacts.

    Return ONE valid JSON object only. No markdown or text outside the JSON.

    PAST ANALYSIS ({age_text}):
    {json.dumps(past_for_prompt, indent=2)}

    ALL HEADLINES:
    {headlines_block}

    NEW HEADLINES:
    {new_headlines_block}

    RULES

    1. PAST ANALYSIS

    Use the past analysis only as context.

    - If it is older than 7 days, ignore it and analyse from scratch.
    - If it is <=1 day old and there are no new headlines, return it unchanged and set updates_from_past_analysis to "None".
    - Otherwise, retain existing analysis unless the headlines provide a reason to change it.

    2. TOP GLOBAL THEMES

    Identify up to {themes_count} important global themes.

    A theme must represent a specific, material story supported by one or more headlines.

    For each theme, write 2-3 sentences:
    - Explain what is happening.
    - Identify the important countries, actors or events involved.
    - Explain the broader economic or market significance only when supported by the headlines.

    Themes should describe actual events in the headlines, not hypothetical future consequences.

    Do not create a theme merely because several headlines share a broad topic.

    3. SECTORS

    Identify up to {sectors_count} affected GICS sectors from:

    Energy, Materials, Industrials, Consumer Discretionary, Consumer Staples, Health Care, Financials, Information Technology, Communication Services, Utilities, Real Estate.

    Only select a sector when the headlines provide a clear, plausible economic mechanism affecting that sector.

    Reason only:

    HEADLINE → ECONOMIC EFFECT → SECTOR IMPACT

    Do NOT reason:

    THEME → SECTOR

    Do not use weak, indirect, speculative or generic relationships.

    If the connection requires multiple assumptions, do not select the sector.

    A headline may have no relevant sector impact.

    It is better to return fewer sectors than weak or speculative sector matches.

    4. SECTOR ANALYSIS

    For each selected sector:

    - associated_theme must exactly match a theme_title.
    - posture must be BUY, SELL, WATCH or AVOID.
    - economic_catalyst must be 2-3 sentences.

    The economic_catalyst must:
    1. State what happened in the headline.
    2. Explain the economic mechanism connecting it to the sector.
    3. Explain why the expected sector impact is positive, negative or uncertain.

    The catalyst must be supported by the supplied headlines.

    Do not introduce new events, policies, spending, demand changes, price changes or other assumptions that are not present or reasonably implied by the headlines.

    Never turn a possible second-order consequence into a catalyst.

    Do not mention individual companies in economic_catalyst.

    BUY = clear evidence that the expected sector impact is positive.
    SELL = clear evidence that the expected sector impact is negative.
    WATCH = the headline has a credible sector impact, but the direction, magnitude or duration is still uncertain.
    AVOID = meaningful downside risk exists, but the evidence is insufficient for a clear SELL.

    The posture describes expected sector performance, not whether the underlying news is good or bad.

    5. COMPANIES

    For each selected sector, select up to {companies_count} real publicly listed companies.

    Only select companies with strong, specific exposure to the identified sector catalyst.

    It is better to return fewer companies than weak or generic matches.

    Format:
    Full Company Name (**TICKER**)

    Never invent companies or tickers.

    6. UPDATES

    Set updates_from_past_analysis to "None" when:
    - there is no past analysis;
    - the past analysis was ignored because it is older than 7 days; or
    - nothing materially changed.

    Otherwise, briefly explain what changed and which headline caused the change.

    FINAL CHECK

    Before responding:

    - Themes describe actual events from the headlines.
    - Every selected sector has a specific headline supporting it.
    - Every sector has a clear economic mechanism linking the headline to the sector.
    - No sector depends on multiple speculative assumptions.
    - Every selected company has strong exposure to the sector catalyst.
    - The posture matches the expected economic impact.
    - Do not manufacture connections to increase the number of themes, sectors or companies.
    - Fewer results are preferable to weak results.
    - associated_theme exactly matches a theme_title.
    - Make sure each (TICKER) in companies is surrounded by double **.
    - Return valid JSON only.

    JSON:

    {{
        "timestamp": "",
        "processed_headlines": "",
        "updates_from_past_analysis": "",
        "top_global_themes": [
            {{
                "theme_title": "",
                "theme_description": ""
            }}
        ],
        "market_opportunities_and_risks": [
            {{
                "impacted_sector": "",
                "associated_theme": "",
                "economic_catalyst": "",
                "posture": "",
                "companies": [ "Company Name (**TICKER**)" ]
            }}
        ]
    }}
    """


def parse_llm_json(raw_text: str) -> dict:
    """Extract the JSON object from the model's reply. Raises ValueError if absent."""
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in the model response.")
    return json.loads(raw_text[start:end], strict=False)


def ensure_ollama_running(startup_timeout: int = 30) -> None:
    """Start the Ollama server if it isn't already running. Raises RuntimeError on failure."""
    if is_ollama_running():
        return

    # Check if the 'ollama' command is available (i.e., Ollama is installed)
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama is not running and the 'ollama' command was not found. Please install it from https://ollama.com/")

    # Hide the console window on Windows; detach from this process on other platforms
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(["ollama", "serve"], **kwargs)

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_ollama_running():
            return
        time.sleep(0.5)

    raise RuntimeError(f"Started Ollama but it did not respond within {startup_timeout} seconds.")


def analyse_headlines(headlines: list[str], settings: Settings) -> dict:
    """Ask the local LLM to analyse the headlines. Raises on failure."""
    headlines_block = "\n".join(f"- {headline}" for headline in headlines)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

    past_analysis = load_past_analysis(settings.output_file, settings.use_previous_analysis)
    age_days = past_analysis_age_days(past_analysis)

    new_headlines = []
    new_headlines_block = "None"

    if settings.use_previous_analysis:
        new_headlines = find_new_headlines(headlines, past_analysis)
        new_headlines_block = "\n".join(f"- {headline}" for headline in new_headlines) or "None"

    system_prompt = build_system_prompt(
        past_analysis,
        age_days,
        headlines_block,
        new_headlines_block,
        settings.global_themes_count,
        settings.affected_sectors_count,
        settings.affected_companies_count,
    )

    response = ollama.chat(
        model=settings.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyse these current global headlines:\n{headlines_block}"},
        ],
        format="json",
        options={"temperature": 0.2},
    )

    result = parse_llm_json(response["message"]["content"])
    result["timestamp"] = timestamp
    result["processed_headlines"] = headlines_block
    return result

# --------------------------------------------------------------------------- #
# 5. UI
# --------------------------------------------------------------------------- #

def render_intro() -> None:
    st.title("📰 AI News Analyser", text_alignment="center")
    st.write(
        "This tool uses a local LLM to analyse the latest global news headlines and identify the "
        "top global themes, as well as the most impacted sectors and companies. It then provides a "
        "recommended posture for each sector based on the macroeconomic situation. It is designed to "
        "help investors and analysts quickly understand the potential market opportunities and risks "
        "arising from current events."
    )
    st.write(
        "Settings are provided in the sidebar, where you can configure many aspects of the tool, "
        "including the RSS feed URL, the number of headlines to analyse, the number of global themes "
        "etc. You can also choose to save the analysis results to a JSON file for later review."
    )
    st.info(
        "The tool is powered by [Ollama](https://ollama.com/), which is an open-source tool that "
        "allows users to run large language models locally and uses the "
        "[BBC World News RSS feed](http://feeds.bbci.co.uk/news/world/rss.xml) as its default "
        "source of headlines.",
        icon="ℹ️",
    )
    st.warning(
        "This tool is for informational purposes only. It does not constitute financial advice. "
        "Please consult a licensed financial advisor before making any investment decisions.",
        icon="⚠️",
    )


def render_sidebar(saved: Settings) -> tuple[Settings, bool]:
    """Draw the settings sidebar. Returns the chosen settings and whether Run was clicked."""
    with st.sidebar:
        st.title("⚙️ Settings")

        with st.expander("**📰 Configure Fetching**", expanded=True):
            feed_url = st.text_input("**RSS Feed URL**", value=saved.feed_url, type="url", on_change="ignore")
            timeout = st.number_input("**Timeout (Seconds)**", min_value=1, max_value=60, value=saved.timeout, step=1, on_change="ignore", help="How long to wait for the RSS feed to respond before giving up.")

        with st.expander("**🔍 Tune Analysis**", expanded=True):
            themes = st.slider("**Max Global Themes**", min_value=1, max_value=5, value=saved.global_themes_count, step=1, on_change="ignore")
            sectors = st.slider("**Max Affected Sectors**", min_value=1, max_value=5, value=saved.affected_sectors_count, step=1, on_change="ignore")
            companies = st.slider("**Max Affected Companies (Per Sector)**", min_value=1, max_value=5, value=saved.affected_companies_count, step=1, on_change="ignore")
            model = st.selectbox("**AI Model**", options=[model.model for model in ollama.list().models], on_change="ignore", help="Choose the local LLM to use for analysis. Only installed models are shown.")
            use_previous_analysis = st.toggle("**Use Previous Analysis**", value=saved.use_previous_analysis, on_change="ignore", help="Whether to use the previously saved analysis as context for the current analysis.")

        with st.expander("**▶️ Run Analysis**", expanded=True):
            auto_run = st.toggle("**Auto Run**", value=saved.auto_run, on_change="ignore", help="Whether to automatically re-run the analysis at regular intervals.")
            interval = st.time_input("**Interval**", value=datetime.strptime(saved.interval, "%H:%M:%S").time(), help="HH:MM format.", on_change="ignore")

        with st.expander("**💾 Format Output**", expanded=True):
            save_analysis = st.toggle("**Save Analysis**", value=saved.save_analysis, on_change="ignore")
            output_file = st.text_input("**Output File Name**", value=saved.output_file, on_change="ignore", help="Saved into the current working directory.")
            json_indent = st.number_input("**Indent Level (JSON)**", min_value=0, max_value=8, value=saved.json_indent, step=1, on_change="ignore")

        run_clicked = st.button("**Run News Analysis**", width="stretch")

    settings = Settings(
        feed_url=feed_url,
        timeout=timeout,
        global_themes_count=themes,
        affected_sectors_count=sectors,
        affected_companies_count=companies,
        model=model,
        use_previous_analysis=use_previous_analysis,
        auto_run=auto_run,
        interval=interval.strftime("%H:%M:%S") if auto_run else saved.interval,
        save_analysis=save_analysis,
        output_file=output_file if save_analysis else saved.output_file,
        json_indent=json_indent if save_analysis else saved.json_indent,
    )
    return settings, run_clicked


def format_posture(posture: str | None) -> str:
    colour = POSTURE_COLOURS.get(posture)
    return f":{colour}[**{posture}**]" if colour else str(posture)


def format_headline(index: int, headline: str) -> str:
    return f"{index}. **{headline.strip()}**"


def format_theme(index: int, theme: dict) -> str:
    return f"{index}. **{theme.get('theme_title')}** - {theme.get('theme_description')}"


def format_opportunity(index: int, item: dict) -> str:
    return (
        f"{index}. **{item.get('impacted_sector')} Sector** ({item.get('associated_theme')}):\n"
        f"    - **Catalyst** - {item.get('economic_catalyst')}\n"
        f"    - **Posture** - {format_posture(item.get('posture'))}\n"
        f"    - **Companies** - {', '.join(item.get('companies', []))}\n"
    )


def render_results(output_container, headlines: list[str], results: dict) -> None:
    with output_container:
        analysis_tab, headlines_tab = st.tabs(["📊 Analysis Results", "📰 Analysed Headlines"])

        headlines_tab.title("📰 Analysed Headlines", text_alignment="center")
        _, col2, _ = headlines_tab.columns([1, 6, 1])
        for index, headline in enumerate(headlines, start=1):
            col2.write(format_headline(index, headline))

        analysis_tab.title("📊 Analysis Results", text_alignment="center")
        analysis_tab.header("Top Global Themes", text_alignment="center")
        for index, theme in enumerate(results.get("top_global_themes", []), start=1):
            analysis_tab.write(format_theme(index, theme))

        analysis_tab.divider()

        analysis_tab.header("Market Opportunities & Risks", text_alignment="center")
        for index, item in enumerate(results.get("market_opportunities_and_risks", []), start=1):
            analysis_tab.write(format_opportunity(index, item))

        if has_updates(results):
            analysis_tab.divider()
            analysis_tab.info(f"**Updates since last analysis:** {results['updates_from_past_analysis']}", icon="🔄")


# --------------------------------------------------------------------------- #
# 6. Orchestration
# --------------------------------------------------------------------------- #

def initialise_ollama() -> None:
    """Ensure Ollama is running, and if not, start it."""

    if "ollama_running" not in st.session_state:
        try:
            ensure_ollama_running()
            st.session_state.ollama_running = is_ollama_running()
        except RuntimeError as e:
            st.error(f"Failed to start Ollama: {e}", icon="❌")
            st.stop()


def run_analysis(settings: Settings) -> None:
    """Fetch, analyse, display and (optionally) save. Each stage reports its own errors."""
    output_container = st.empty()
    status_container = st.empty()

    while True:
        status_container.empty()
        output_container.empty()
        centered_container = status_container.container(horizontal_alignment="center")

        try:
            with centered_container.spinner(f"Gathering news from {settings.feed_url}, please wait..."):
                time.sleep(2.0)
                headlines = fetch_headlines(settings.feed_url, settings.timeout)
        except Exception as e:
            status_container.error(f"Error fetching news: {e}", icon="❌")
            return

        if not headlines:
            status_container.warning("No headlines were found in that feed.", icon="⚠️")
            return

        try:
            headline_count = "headline" if len(headlines) == 1 else "headlines"
            with centered_container.spinner(f"Processed {len(headlines)} {headline_count}. Running headline analyser, please wait...", show_time=True):
                results = analyse_headlines(headlines, settings)
        except Exception as e:
            status_container.error(f"Error analysing headlines: {e}", icon="❌")
            return

        render_results(output_container, headlines, results)

        if settings.save_analysis:
            try:
                path = save_analysis_to_disk(results, settings.output_file, settings.json_indent)
                st.toast(f"Saved analysis to {path}", icon="💾")
            except Exception as e:
                status_container.error(f"Error saving analysis to disk: {e}", icon="❌")

        if not settings.auto_run:
            break

        total_seconds = get_total_seconds_from_time(datetime.strptime(settings.interval, "%H:%M:%S").time())
        status_container.info(f"Auto-run is enabled. Next analysis will run in {settings.interval}.", icon="⏳")
        time.sleep(total_seconds)


def main() -> None:
    st.set_page_config(page_title="AI News Analyser", page_icon="📰")
    st.html(PAGE_CSS)
    initialise_ollama()
    render_intro()

    settings, run_clicked = render_sidebar(Settings.load())

    if run_clicked:
        settings.save()
        run_analysis(settings)


if __name__ == "__main__":
    main()