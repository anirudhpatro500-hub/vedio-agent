# 🎬 Video+Agent — AI Meeting Assistant

Converts YouTube videos into structured meeting insights using AI.

## Pipeline

1. **Audio Extraction** — Downloads YouTube audio via `yt-dlp`
2. **Speech-to-Text** — Local Whisper transcription (or Sarvam API for Hindi)
3. **LLM Analysis** — Gemini via OpenRouter
   - Title generation
   - Summary (map-reduce)
   - Key decisions
   - Action items
   - Open questions
4. **RAG Q&A** — ChromaDB vector store + LangChain for meeting chat

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio | yt-dlp, FFmpeg, PyDub |
| STT | OpenAI Whisper (local) / Sarvam AI |
| LLM | Gemini 2.5 Flash (via OpenRouter) |
| RAG | ChromaDB, Sentence-Transformers, LangChain |
| UI | Streamlit / Rich (terminal) |

## Setup

```bash
uv venv --python 3.13
.venv\Scripts\activate
uv pip install -r requirements.txt   

python main.py
# or
streamlit run app.py   

├── app.py              # Streamlit UI
├── main.py             # Terminal CLI
├── core/
│   ├── transcriber.py  # Whisper / Sarvam STT
│   ├── summarize.py    # LLM summarization
│   ├── extractor.py    # Decisions, actions, questions
│   ├── rag_engine.py   # RAG Q&A chain
│   └── vector_store.py # ChromaDB setup
├── utils/
│   └── audio_processor.py  # Download, convert, chunk
└── requirements.txt   

