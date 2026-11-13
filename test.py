from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

console = Console()

source = "https://www.youtube.com/watch?v=re9r0kzQp_M"
language = "english"

# ─── STEP 1: Audio Processing ─────────────────────────────
console.print(Panel.fit("[bold cyan]🎬 Video+Agent — AI Meeting Assistant[/bold cyan]", border_style="cyan"))
console.print(Rule("[bold]Step 1: Audio Processing[/bold]"))
console.print("  [dim]Source:[/dim] " + source)
chunks = process_input(source)

# ─── STEP 2: Transcription ────────────────────────────────
console.print(Rule("[bold]Step 2: Speech-to-Text (Whisper)[/bold]"))
transcript = transcribe_all(chunks, language)
console.print(Panel(transcript[:600] + ("..." if len(transcript) > 600 else ""), title="Transcript (first 600 chars)", border_style="blue"))

# ─── STEP 3: LLM Analysis ─────────────────────────────────
console.print(Rule("[bold]Step 3: LLM Analysis (Gemini via OpenRouter)[/bold]"))

title = generate_title(transcript)
console.print(Panel(f"[bold]{title}[/bold]", title="Title", border_style="magenta"))

summary = summarize(transcript)
console.print(Panel(summary, title="Summary", border_style="green"))

decisions = extract_key_decisions(transcript)
console.print(Panel(decisions, title="Key Decisions", border_style="yellow"))

actions = extract_action_items(transcript)
console.print(Panel(actions, title="Action Items", border_style="yellow"))

questions = extract_questions(transcript)
console.print(Panel(questions, title="Open Questions", border_style="yellow"))

console.print(Panel.fit("[bold green]✅ Done[/bold green]", border_style="green"))   