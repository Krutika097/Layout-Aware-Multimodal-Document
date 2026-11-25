
import base64, io, os, re
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pdfplumber, pymupdf, requests, streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA="http://localhost:11434"
CHAT_MODEL="llama3.2:3b"
EMBED_MODEL="nomic-embed-text"
VISION_MODEL="llava:7b"
TOP_K=6

st.set_page_config(page_title="Layout-Aware Multimodal Document Intelligence", page_icon="📄", layout="wide")
st.title("📄 Layout-Aware Multimodal Document Intelligence")
st.caption("Version 3 · OCR · local AI Q&A · layout · tables · images · relationship graph · visual evidence")

for k,v in {"documents":{},"chunks":[],"last_answer":None}.items():
    if k not in st.session_state: st.session_state[k]=v

def clean(s):
    s=(s or "").replace("\x00"," ")
    s=re.sub(r"(?<=\w)-\s+(?=[a-z])","",s)
    s=re.sub(r"[ \t]+"," ",s)
    return s.strip()

def split_chunks(text, size=180, overlap=30):
    words=text.split(); out=[]; start=0
    while start<len(words):
        end=min(start+size,len(words)); part=" ".join(words[start:end]).strip()
        if part: out.append(part)
        if end==len(words): break
        start=max(end-overlap,start+1)
    return out

def ollama():
    try:
        r=requests.get(f"{OLLAMA}/api/tags",timeout=4); r.raise_for_status()
        return True,[m.get("name","") for m in r.json().get("models",[])]
    except requests.RequestException: return False,[]

def has_model(models,name):
    return any(x==name or x.startswith(name+":") for x in models)

def embeddings(texts):
    out=[]
    for i in range(0,len(texts),24):
        r=requests.post(f"{OLLAMA}/api/embed",json={"model":EMBED_MODEL,"input":texts[i:i+24]},timeout=300)
        r.raise_for_status(); out.extend(r.json()["embeddings"])
    return out

def tesseract_path():
    for p in [os.getenv("TESSERACT_CMD"),r"C:\Program Files\Tesseract-OCR\tesseract.exe",r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if p and Path(p).is_file(): return p
    return None

def ocr_page(page):
    import pytesseract
    from pytesseract import Output
    cmd=tesseract_path()
    if cmd: pytesseract.pytesseract.tesseract_cmd=cmd
    scale=2.2
    pix=page.get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False)
    image=Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    d=pytesseract.image_to_data(image,lang="eng",output_type=Output.DICT,config="--psm 3")
    lines={}
    for i,w in enumerate(d["text"]):
        w=clean(w)
        try: conf=float(d["conf"][i])
        except: conf=-1
        if not w or conf<25: continue
        key=(d["block_num"][i],d["par_num"][i],d["line_num"][i])
        lines.setdefault(key,[]).append((w,d["left"][i]/scale,d["top"][i]/scale,(d["left"][i]+d["width"][i])/scale,(d["top"][i]+d["height"][i])/scale))
    result=[]
    for words in lines.values():
        result.append({"text":" ".join(x[0] for x in words),"bbox":[round(min(x[1] for x in words),1),round(min(x[2] for x in words),1),round(max(x[3] for x in words),1),round(max(x[4] for x in words),1)],"kind":"ocr_text"})
    return result

def get_tables(page,pdf_bytes,index):
    found=[]
    try:
        ft=page.find_tables()
        if ft and ft.tables: found=[t.extract() for t in ft.tables if t.extract()]
    except: pass
    if not found:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf: found=pdf.pages[index].extract_tables() or []
        except: pass
    result=[]
    for table in found:
        rows=[[clean(c) for c in (row or [])] for row in table]
        if any(any(c for c in row) for row in rows): result.append(rows)
    return result

def parse_pdf(data,name,ocr_mode):
    pdf=pymupdf.open(stream=data,filetype="pdf")
    doc={"name":name,"pages":[],"tables":[],"images":[],"chunks":[],"total_pages":len(pdf),"ocr_pages":0,"ocr_errors":[],"pdf_bytes":data}
    for idx,page in enumerate(pdf):
        pno=idx+1; blocks=[]
        for b in page.get_text("blocks",sort=True):
            if len(b)>=5 and clean(b[4]):
                blocks.append({"text":clean(b[4]),"bbox":[round(float(x),1) for x in b[:4]],"kind":"native_text"})
        chars=sum(len(b["text"]) for b in blocks)
        should_ocr=ocr_mode=="All pages" or (ocr_mode=="Scanned pages only" and chars<40)
        if should_ocr:
            try:
                ocr=ocr_page(page)
                if ocr:
                    doc["ocr_pages"]+=1
                    blocks=(blocks+ocr) if ocr_mode=="All pages" else ocr
            except Exception as e: doc["ocr_errors"].append(f"Page {pno}: {type(e).__name__}: {e}")
        doc["pages"].append({"page":pno,"width":round(page.rect.width,1),"height":round(page.rect.height,1),"blocks":blocks,"ocr_used":any(b["kind"]=="ocr_text" for b in blocks)})
        for b in blocks:
            for part in split_chunks(b["text"]):
                doc["chunks"].append({"text":part,"source":name,"page":pno,"bbox":b["bbox"],"kind":b["kind"],"embedding":None})
        for ti,rows in enumerate(get_tables(page,data,idx),1):
            doc["tables"].append({"page":pno,"table_number":ti,"rows":rows})
            for part in split_chunks("\n".join(" | ".join(row) for row in rows)):
                doc["chunks"].append({"text":part,"source":name,"page":pno,"bbox":None,"kind":"table","embedding":None})
        for ii,info in enumerate(page.get_images(full=True),1):
            try:
                ex=pdf.extract_image(info[0])
                doc["images"].append({"name":f"{Path(name).stem}_p{pno}_img{ii}.{ex.get('ext','png')}","page":pno,"bytes":ex["image"]})
            except: pass
    pdf.close()
    return doc

def search_semantic(q,chunks):
    usable=[c for c in chunks if c.get("embedding") is not None]
    if not usable:return []
    qv=np.asarray(embeddings([q])[0],dtype=np.float32)
    mat=np.asarray([c["embedding"] for c in usable],dtype=np.float32)
    scores=(mat@qv)/((np.linalg.norm(mat,axis=1)+1e-12)*(np.linalg.norm(qv)+1e-12))
    return [(usable[i],float(scores[i])) for i in np.argsort(scores)[::-1][:TOP_K]]

def search_lexical(q,chunks):
    usable=[c for c in chunks if c.get("text")]
    try:
        v=TfidfVectorizer(stop_words="english",ngram_range=(1,2))
        m=v.fit_transform([c["text"] for c in usable]+[q]); scores=cosine_similarity(m[-1],m[:-1]).flatten()
        return [(usable[i],float(scores[i])) for i in np.argsort(scores)[::-1][:TOP_K] if scores[i]>0]
    except: return []

def answer(q,evidence):
    context="\n\n".join(f"[{c['source']}, page {c['page']}, {c['kind']}]\n{c['text']}" for c,_ in evidence)
    prompt=f"Answer using ONLY this document evidence. Do not invent facts, names, numbers or citations. Say if evidence is insufficient. Cite page numbers.\nQUESTION: {q}\nEVIDENCE:\n{context}"
    r=requests.post(f"{OLLAMA}/api/chat",json={"model":CHAT_MODEL,"messages":[{"role":"system","content":"You are a careful document research assistant. Never fabricate."},{"role":"user","content":prompt}],"stream":False,"options":{"temperature":0.1}},timeout=300)
    r.raise_for_status(); return r.json()["message"]["content"].strip()

def graph_for(docs):
    g=nx.Graph()
    for name,doc in docs.items():
        dn=f"Document: {name}"; g.add_node(dn,kind="document")
        for p in doc["pages"]:
            pn=f"{name}\nPage {p['page']}"; g.add_node(pn,kind="page"); g.add_edge(dn,pn)
            for b in p["blocks"]:
                t=b["text"].strip()
                if 8<len(t)<110 or re.match(r"^(chapter|section|\d+(\.\d+)*\s)",t,re.I):
                    hn="Heading: "+t[:65]; g.add_node(hn,kind="heading"); g.add_edge(pn,hn)
        for t in doc["tables"]:
            n=f"Table {t['table_number']} · Page {t['page']}"; g.add_node(n,kind="table"); g.add_edge(dn,n)
        for i,img in enumerate(doc["images"],1):
            n=f"Image {i} · Page {img['page']}"; g.add_node(n,kind="image"); g.add_edge(dn,n)
    return g

def highlighted_page(name,page_no,bbox):
    pdf=pymupdf.open(stream=st.session_state.documents[name]["pdf_bytes"],filetype="pdf")
    page=pdf[page_no-1]
    if bbox: page.draw_rect(pymupdf.Rect(*bbox),color=(1,0,0),width=2,overlay=True)
    pix=page.get_pixmap(matrix=pymupdf.Matrix(1.35,1.35),alpha=False)
    image=Image.open(io.BytesIO(pix.tobytes("png"))); pdf.close(); return image

def vision_analysis(image_bytes):
    _,models=ollama()
    if not has_model(models,VISION_MODEL):
        raise RuntimeError(f"Optional vision model missing. Run: ollama pull {VISION_MODEL}")
    encoded=base64.b64encode(image_bytes).decode()
    r=requests.post(f"{OLLAMA}/api/chat",json={"model":VISION_MODEL,"messages":[{"role":"user","content":"Analyze this chart, figure or diagram. Identify type, labels, axes, legend, trends and visible values. Do not guess unreadable information.","images":[encoded]}],"stream":False},timeout=300)
    r.raise_for_status(); return r.json()["message"]["content"]

running,models=ollama()
with st.sidebar:
    st.header("1. Local AI status")
    st.success("Ollama is running") if running else st.error("Open Ollama desktop")
    st.caption("Installed models:")
    for m in models: st.write("• "+m)
    st.caption(f"Chat: {CHAT_MODEL} · Embeddings: {EMBED_MODEL}")
    st.divider(); st.header("2. Upload documents")
    uploads=st.file_uploader("Choose PDF files",type=["pdf"],accept_multiple_files=True)
    ocr_mode=st.selectbox("OCR mode",["Scanned pages only","All pages","OCR disabled"],help="Scanned pages only OCRs pages with little selectable text.")
    process=st.button("Process documents",type="primary",use_container_width=True)
    st.divider()
    for item in ["PDF text and layout","OCR for scanned pages","Page coordinates","Table extraction","Embedded image extraction","Local semantic search","Local LLM answers with evidence","Document relationship graph","Visual evidence highlighting","Optional local chart/diagram AI"]:
        st.write("✓ "+item)

if process:
    if not uploads: st.warning("Upload at least one PDF.")
    elif not running: st.error("Start Ollama first.")
    else:
        st.session_state.documents={}; st.session_state.chunks=[]; st.session_state.last_answer=None
        bar=st.progress(0); status=st.empty()
        for i,up in enumerate(uploads):
            try:
                status.write(f"Extracting PDF, layout and OCR: {up.name}")
                doc=parse_pdf(up.getvalue(),up.name,ocr_mode)
                status.write(f"Creating embeddings for {len(doc['chunks'])} passages: {up.name}")
                if doc["chunks"]:
                    vectors=embeddings([c["text"] for c in doc["chunks"]])
                    for c,v in zip(doc["chunks"],vectors): c["embedding"]=v
                st.session_state.documents[up.name]=doc; st.session_state.chunks.extend(doc["chunks"])
                for err in doc["ocr_errors"]: st.warning(err)
            except Exception as e: st.error(f"Could not process {up.name}: {e}")
            bar.progress((i+1)/len(uploads))
        status.success("Processing finished.")

if st.session_state.documents:
    docs=st.session_state.documents
    pages=sum(d["total_pages"] for d in docs.values()); tables=sum(len(d["tables"]) for d in docs.values())
    images=sum(len(d["images"]) for d in docs.values()); ocrpages=sum(d["ocr_pages"] for d in docs.values())
    cols=st.columns(5)
    for col,label,value in zip(cols,["Documents","Pages","Tables","Images","OCR pages"],[len(docs),pages,tables,images,ocrpages]): col.metric(label,value)
    qa,layout,tabtables,tabimages,graph,evidence=st.tabs(["💬 AI document Q&A","🧩 Layout and text","📊 Extracted tables","🖼️ Images and charts","🕸️ Relationship graph","🔎 Visual evidence"])
    with qa:
        st.subheader("Ask your documents")
        q=st.text_input("Your question",placeholder="What datasets are discussed in this paper?")
        if st.button("Generate AI answer",type="primary",disabled=not running):
            if not q.strip(): st.warning("Enter a question.")
            else:
                with st.spinner("Searching and generating local answer..."):
                    try:
                        ev=search_semantic(q,st.session_state.chunks) or search_lexical(q,st.session_state.chunks)
                        if ev: st.session_state.last_answer={"question":q,"answer":answer(q,ev),"evidence":ev}
                        else: st.warning("No relevant passages found.")
                    except Exception as e: st.error(f"AI request failed: {e}")
        if st.session_state.last_answer:
            st.markdown("### AI-generated answer"); st.markdown(st.session_state.last_answer["answer"])
            st.markdown("### Supporting evidence")
            for c,s in st.session_state.last_answer["evidence"]:
                with st.container(border=True):
                    st.markdown(f"**{c['source']} — Page {c['page']}**"); st.caption(f"{c['kind']} · similarity {s:.3f}"); st.write(c["text"])
                    if c["bbox"]: st.caption(f"Coordinates: {c['bbox']}")
    with layout:
        name=st.selectbox("Document",list(docs),key="layout_doc")
        pno=st.selectbox("Page",list(range(1,docs[name]["total_pages"]+1)),key="layout_page")
        p=docs[name]["pages"][pno-1]
        st.write(f"Page dimensions: {pymupdf.Rect(0,0,p['width'],p['height'])} · OCR used: {p['ocr_used']}")
        for i,b in enumerate(p["blocks"],1):
            with st.container(border=True):
                st.caption(f"Block {i} · {b['kind']} · bbox {b['bbox']}"); st.write(b["text"])
        if not p["blocks"]: st.info("No text found. Check Tesseract installation or OCR language.")
    with tabtables:
        count=0
        for name,d in docs.items():
            for t in d["tables"]:
                count+=1; st.markdown(f"**{name} — Page {t['page']} — Table {t['table_number']}**"); st.dataframe(t["rows"],use_container_width=True)
        if not count: st.info("No tables detected. Complex or image-based tables may need specialized recognition.")
    with tabimages:
        st.info(f"Optional image understanding model: ollama pull {VISION_MODEL}. Image extraction and the rest of the app work without it.")
        for name,d in docs.items():
            for i,img in enumerate(d["images"]):
                with st.container(border=True):
                    st.markdown(f"**{name} — Page {img['page']} — {img['name']}**"); st.image(img["bytes"],width=450)
                    if st.button("Analyze with local vision AI",key=f"vis_{name}_{i}"):
                        with st.spinner("Analyzing image..."):
                            try: st.session_state[f"visres_{name}_{i}"]=vision_analysis(img["bytes"])
                            except Exception as e: st.error(str(e))
                    if st.session_state.get(f"visres_{name}_{i}"): st.write(st.session_state[f"visres_{name}_{i}"])
        if not images: st.info("No embedded images found.")
    with graph:
        g=graph_for(docs); st.write(f"{g.number_of_nodes()} nodes · {g.number_of_edges()} relationships")
        if g.number_of_nodes():
            fig,ax=plt.subplots(figsize=(13,8)); pos=nx.spring_layout(g,seed=42)
            colors=[{"document":"#2563eb","page":"#16a34a","heading":"#f59e0b","table":"#9333ea","image":"#db2777"}.get(g.nodes[n].get("kind"),"#64748b") for n in g.nodes]
            nx.draw_networkx(g,pos,ax=ax,node_color=colors,node_size=800,font_size=7,with_labels=True,edge_color="#999")
            ax.set_axis_off(); st.pyplot(fig,use_container_width=True); plt.close(fig)
        st.caption("This first graph links documents, pages, detected headings, tables and images. Semantic entity relationships are a future enhancement.")
    with evidence:
        if st.session_state.last_answer:
            ev=st.session_state.last_answer["evidence"]
            opts=[f"{c['source']} · Page {c['page']} · {c['kind']}" for c,s in ev]
            ix=st.selectbox("Evidence passage",range(len(opts)),format_func=lambda i:opts[i])
            c,s=ev[ix]; st.write(c["text"]); st.caption(f"Retrieval score {s:.3f}")
            if c["bbox"]:
                st.image(highlighted_page(c["source"],c["page"],c["bbox"]),caption="Retrieved region highlighted on original PDF page",use_container_width=True)
            else: st.info("This passage has no text bounding box (for example, a table passage).")
        else: st.info("Ask a question in AI document Q&A first.")
else:
    st.info("Upload a PDF, select an OCR mode and click Process documents.")
    st.write("PDF → native text/OCR → layout/tables/images → local embeddings → retrieval → local AI answer with evidence.")
