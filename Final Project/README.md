# IITB Insti-Assist — Academic Assistant (RAG)

A Retrieval-Augmented Generation assistant that answers questions about IIT Bombay
**academics** — registration, grading (SPI/CPI), the academic calendar, medals &
prizes, academic malpractice rules, and PG/M.Sc. regulations — grounded strictly in
11 real, official IITB documents. It refuses to answer ("I don't know...") when a
question falls outside those documents.

## 1. Scope

**Academic Assistant** track: course registration, grading policy, academic
calendar, exam rules, PG regulations, medals/prizes, and academic malpractice
consequences. (Deliberately excludes hostel life, clubs, and general campus
topics — see "Known Limitations" below.)

## 2. Data sources

All 11 documents are official PDFs published by the IIT Bombay Academic Office,
fetched directly and converted to structured plain-text summaries for chunking
(`data/raw/`):

| File | Source |
|---|---|
| `01_ug_rule_book.txt` | [UG Rules & Regulations, updated June 2025](https://acad.iitb.ac.in/files/UG_RULE_BOOK.pdf) |
| `02_academic_calendar_2025_26.txt` | [Academic Calendar 2025-26](https://acad.iitb.ac.in/sites/default/files/Academic%20Calendar%202025-26_FINAL.pdf) |
| `03_medals_and_academic_prizes.txt` | [Rules for Award of Medals & Academic Prizes](https://www.iitb.ac.in/newacadhome/RulesforAwardofMedalsandAcademicprizesforUGandPG.pdf) |
| `04_academic_malpractice_punishments.txt` | [Disciplinary Actions for Academic Malpractice](https://www.iitb.ac.in/newacadhome/punishments201521July.pdf) |
| `05_msc_pg_rules.txt` | [M.Sc./PG Rules & Regulations](https://acad.iitb.ac.in/files/M.Sc_.%20Rules.pdf) |
| `06_phd_rules.txt` | [Ph.D. Programme Rules & Regulations, July 2025](https://acad.iitb.ac.in/files/Ph.D.%20Rules.pdf) |
| `07_mtech_mdes_mba_mpp_rules.txt` | [M.Tech./MPP/M.Des./MBA Rules & Regulations, July 2025](https://acad.iitb.ac.in/files/M.Tech_.%20MPP.%20M.Des_.%20MBA%20Rules_0.pdf) |
| `08_exit_degree_rules.txt` | [Exit Degree Rules (M.Tech./MPP/MS by Research/Dual Degree/PhD)](https://acad.iitb.ac.in/files/ExitDegreemtechphd_0.pdf) |
| `09_scholarships_faq.txt` | [Institute Scholarships FAQ (UG & M.Sc.)](https://www.iitb.ac.in/newacadhome/FAQ_Scholarship.pdf) |
| `10_ta_ra_fa_duty_guidelines.txt` | [Guidelines for TA/RA/TAP/RAP/FA Duty](https://www.iitb.ac.in/newacadhome/TA-RADuty.pdf) |
| `11_convocation_procedure.txt` | [Convocation Form Submission Procedure](https://www.iitb.ac.in/newacadhome/ConvocationProcedure.pdf) |

## 3. Chunking strategy

`ingest.py` splits each document on paragraph boundaries (blank lines) and
packs paragraphs into ~1000-character chunks with a 200-character overlap,
carried over from the tail of the previous chunk. This keeps each chunk close
to a single rule/topic (e.g. one grading rule, one registration deadline)
rather than cutting a rule in half, while the overlap keeps continuity when a
rule does span a chunk boundary. Long paragraphs that exceed the chunk size on
their own are hard-split with the same overlap. This produced 97 chunks
across the 11 source documents.

## 4. Architecture

```
User question
     │
     ▼
Embed query (sentence-transformers: all-MiniLM-L6-v2)
     │
     ▼
FAISS similarity search → top-4 chunks (cosine similarity via inner product)
     │
     ▼
If best similarity < 0.30 → return "I don't know" immediately (no LLM call)
     │
     ▼
Inject retrieved chunks + question into a strict system prompt
     │
     ▼
Groq API (Llama 3.3 70B) generates an answer using ONLY the provided context
     │
     ▼
Streamlit UI displays the answer + an expandable "Sources used" panel
```

## 5. Setup & Run

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows (PowerShell): .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the vector index (downloads the embedding model on first run,
#    needs internet access to huggingface.co)
python ingest.py

# 4. Set your Groq API key (free — get one at https://console.groq.com/keys)
#    or paste it into the sidebar at runtime instead
export GROQ_API_KEY="gsk_..."          # PowerShell: $env:GROQ_API_KEY="gsk_..."

# 5. Launch the app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

Note: the virtual environment only needs to be created once (step 1). On future runs in a new terminal, you just need to re-activate it (`source venv/bin/activate` or the PowerShell equivalent) before steps 4-5 — no need to repeat `pip install` or `python -m venv`.

## 6. Example questions to try

- "What is the minimum CPI to avoid being sent to the Academic Rehabilitation Programme?"
- "How many credits does a Minor require?"
- "What happens if I'm caught using a mobile phone during an exam?"
- "When does the Spring 2025-26 semester end?"
- "What's the eligibility CPI for the President of India Medal?"
- "How many hours per week does a TA have to work?"
- "What happens if my PhD thesis is rejected by both external referees?"
- "What is the hostel curfew time?" → should trigger "I don't know" (out of scope)

## 7. Known limitations / what I'd improve with more time

- **Coverage**: only 11 documents / academic-office-level rules are indexed.
  Department-specific curricula, minor/honours course lists, and FAQ-style
  student forum content are not included, so department-level questions will
  correctly return "I don't know."
- **Chunking**: paragraph-based chunking is simple and works well for prose
  but is not perfectly tuned for tabular content (e.g. the full academic
  calendar's date tables) — a table-aware chunker would preserve row/column
  relationships more faithfully.
- **Retrieval**: uses a single dense embedding retriever (FAISS + MiniLM).
  A hybrid approach (dense + BM25 keyword search) would likely improve
  recall on queries that use exact terms/codes (e.g. "GC 101", "DX grade").
- **Evaluation**: no formal retrieval/answer quality evaluation set was built
  due to time constraints; a small labeled Q&A set would let us measure
  precision of the "I don't know" refusal behavior.
- **Multi-turn memory**: not implemented (bonus goal) — each question is
  currently answered independently, without conversation history.
- **Citation highlighting**: currently shows the source document name and a
  relevance score, not the exact quoted sentence within the chunk (bonus goal).

## 8. Project structure

```
iitb-insti-assist/
├── data/
│   ├── raw/            # 11 source documents (plain text, from official PDFs)
│   └── index/          # generated: faiss.index, chunks.json
├── ingest.py           # chunking + embedding + index building
├── rag.py              # retrieval + grounded generation logic
├── app.py              # Streamlit UI
├── requirements.txt
└── README.md
```
