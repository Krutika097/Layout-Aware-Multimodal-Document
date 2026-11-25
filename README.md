# Layout-Aware Multimodal Document Intelligence — Version 3

## 1. Install Tesseract OCR (required for OCR)
- Download the Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
- Install it. Default expected path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Restart VS Code after installation.
- If installed elsewhere, in PowerShell set:
  `$env:TESSERACT_CMD = "D:\Apps\Tesseract-OCR\tesseract.exe"`

`pytesseract` is a Python connector; the Tesseract OCR application must also be installed.

## 2. Replace the Version 2 files
Back up the working Version 2 folder. Extract this ZIP. Copy `app.py` and `requirements.txt` into the folder containing your working Version 2 `app.py`, replacing those two files. Keep your existing `.venv`.

## 3. Install dependencies in VS Code terminal
Activate the same virtual environment you used for Version 2. If `.venv` is inside this project:
```powershell
.\.venv\Scripts\Activate.ps1
```
If `.venv` is in its parent folder:
```powershell
..\.venv\Scripts\Activate.ps1
```
Then:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Confirm Ollama models
```powershell
ollama list
```
Required: `llama3.2:3b` and `nomic-embed-text`.

Optional local image understanding:
```powershell
ollama pull llava:7b
```
This model is several GB and may be slow on PCs with limited RAM. Skip it if necessary; OCR, text Q&A, layout, tables, graph and evidence highlighting still work.

## 5. Run
From the folder containing `app.py`:
```powershell
python -m streamlit run app.py
```
Open http://localhost:8501.

## 6. Use the app
Upload PDF, choose:
- `Scanned pages only` (recommended): OCR pages with almost no selectable text.
- `All pages`: OCR every page; slower and may produce duplicate text.
- `OCR disabled`: native text extraction only.
Click Process documents. Tabs provide Q&A, layout, tables, extracted images, relationship graph and visual evidence.

## Troubleshooting
- `TesseractNotFoundError`: install Tesseract app; verify the executable path; restart VS Code. Set `$env:TESSERACT_CMD` if needed.
- `No module named pytesseract`: activate the right `.venv`; rerun `pip install -r requirements.txt`.
- Ollama connection refused: open Ollama desktop.
- Missing required models: `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`.
- Vision model missing: optional `ollama pull llava:7b`.
- OCR is currently English (`eng`). Other languages require Tesseract language data and changing `lang="eng"` in `ocr_page`.

## Features and limitations
- Preserves Version 2: PDF text/layout, tables, images, local embeddings, semantic retrieval and Llama evidence-based Q&A.
- Adds Tesseract OCR for scanned pages, structural document graph and PDF page rendering with retrieved text-block highlighting.
- The graph currently links documents, pages, headings, tables and images; it is not yet a full semantic knowledge graph.
- Optional chart/diagram analysis uses local LLaVA 7B; quality varies and values must be checked against the original.
- Table extraction remains heuristic. Image-only tables may not be detected as structured tables.
- OCR bounding boxes highlight text regions, not individual words.
- LLM answers can be wrong; always inspect the supporting evidence.
