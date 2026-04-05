import os
import io
import re
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langchain_community.document_loaders import PyMuPDFLoader
from docx import Document as DocxDocument

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


# =========================================================
# LOAD ENV
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BACKEND_URL = "http://127.0.0.1:8000"


# =========================================================
# SESSION STATE
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "generated_pdf_bytes" not in st.session_state:
    st.session_state.generated_pdf_bytes = None

if "generated_result" not in st.session_state:
    st.session_state.generated_result = None

if "generated_question_type" not in st.session_state:
    st.session_state.generated_question_type = None


# =========================================================
# HELPERS
# =========================================================
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_website_inputs(raw: str):
    if not raw or not raw.strip():
        return []

    normalized = raw.replace("||", "\n")
    normalized = normalized.replace(";", "\n").replace(",", "\n")

    items = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue

        if "|" in line:
            label, url = line.split("|", 1)
            label, url = label.strip(), url.strip()
        else:
            label, url = "", line

        if url:
            items.append((label, url))

    seen = set()
    out = []
    for label, url in items:
        if url not in seen:
            out.append((label, url))
            seen.add(url)

    return out


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        text = "\n\n".join(p.page_content for p in pages)
        return clean_text(text)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def extract_text_from_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return clean_text("CSV CONTENT:\n" + df.to_string(index=False))


def extract_text_from_excel(file_bytes: bytes, filename: str) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    parts = [f"EXCEL FILE: {filename}"]
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        parts.append(f"\nSHEET: {sheet}\n" + df.to_string(index=False))
    return clean_text("\n".join(parts))


def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        doc = DocxDocument(path)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return clean_text("DOCX CONTENT:\n" + "\n".join(paras))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = str(file_bytes)
    return clean_text(f"TEXT FILE ({filename}):\n{text}")


def extract_text_from_website(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = clean_text(text)

    max_chars = 120000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"
    return f"WEBSITE CONTENT ({url}):\n{text}"


@st.cache_data(show_spinner=False)
def cached_extract_website(url: str) -> str:
    return extract_text_from_website(url)


@st.cache_data(show_spinner=False)
def cached_extract_pdf(file_bytes: bytes) -> str:
    return extract_text_from_pdf(file_bytes)


@st.cache_data(show_spinner=False)
def cached_extract_csv(file_bytes: bytes) -> str:
    return extract_text_from_csv(file_bytes)


@st.cache_data(show_spinner=False)
def cached_extract_excel(file_bytes: bytes, filename: str) -> str:
    return extract_text_from_excel(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_extract_docx(file_bytes: bytes) -> str:
    return extract_text_from_docx(file_bytes)


@st.cache_data(show_spinner=False)
def cached_extract_txt(file_bytes: bytes, filename: str) -> str:
    return extract_text_from_txt(file_bytes, filename)


def build_sources_block(sources, max_chars_per_source=20000):
    parts = []
    for s in sources:
        txt = clean_text(s.get("text", ""))
        if len(txt) > max_chars_per_source:
            txt = txt[:max_chars_per_source] + "\n\n[TRUNCATED]"
        parts.append(f"[{s['id']}] {s['name']} ({s['type']}):\n{txt}")
    return "\n\n---\n\n".join(parts)


def make_pdf_bytes(title: str, body: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin = 40
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, title)
    y -= 30

    c.setFont("Helvetica", 10)
    max_width = width - 2 * x_margin

    for para in body.split("\n"):
        words = para.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 10) <= max_width:
                line = test
            else:
                if y < 60:
                    c.showPage()
                    c.setFont("Helvetica", 10)
                    y = height - 60
                c.drawString(x_margin, y, line)
                y -= 14
                line = w

        if line.strip():
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 60
            c.drawString(x_margin, y, line)
            y -= 14

        y -= 6

    c.save()
    buffer.seek(0)
    return buffer.read()


@st.cache_resource
def get_vector_store():
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name="questions_generation_tool",
        embedding_function=embedding_model,
        persist_directory="chroma_db",
    )
    return vector_db, embedding_model


# =========================================================
# AUTH FUNCTIONS
# =========================================================
def signup_ui():
    st.subheader("Create Account")

    username = st.text_input("Username", key="signup_username")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")

    if st.button("Sign Up"):
        if not username.strip() or not email.strip() or not password.strip():
            st.warning("Please fill all fields.")
            return

        try:
            response = requests.post(
                f"{BACKEND_URL}/signup",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                },
                timeout=15
            )

            if response.status_code == 200:
                st.success("Account created successfully. Please login.")
            else:
                try:
                    st.error(response.json().get("detail", "Signup failed"))
                except Exception:
                    st.error("Signup failed")
        except Exception as e:
            st.error(f"Backend connection error: {e}")


def login_ui():
    st.subheader("Login")

    identifier = st.text_input("Username or Email", key="login_identifier")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if not identifier.strip() or not password.strip():
            st.warning("Please fill all fields.")
            return

        try:
            response = requests.post(
                f"{BACKEND_URL}/login",
                json={
                    "identifier": identifier,
                    "password": password
                },
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.logged_in = True
                st.session_state.user = data["user"]
                st.success("Login successful")
                st.rerun()
            else:
                try:
                    st.error(response.json().get("detail", "Login failed"))
                except Exception:
                    st.error("Login failed")
        except Exception as e:
            st.error(f"Backend connection error: {e}")


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.generated_pdf_bytes = None
    st.session_state.generated_result = None
    st.session_state.generated_question_type = None
    st.rerun()


def save_pdf_to_backend():
    if not st.session_state.generated_pdf_bytes:
        st.warning("Please generate a PDF first.")
        return

    if not st.session_state.user:
        st.warning("Please login first.")
        return

    try:
        files = {
            "pdf_file": ("exam_questions.pdf", st.session_state.generated_pdf_bytes, "application/pdf")
        }
        data = {
            "user_id": str(st.session_state.user["id"]),
            "question_type": st.session_state.generated_question_type or "Unknown"
        }

        response = requests.post(
            f"{BACKEND_URL}/save-pdf",
            data=data,
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            st.success("PDF saved successfully to backend database.")
            st.json(response.json())
        else:
            try:
                st.error(response.json().get("detail", "Failed to save PDF"))
            except Exception:
                st.error("Failed to save PDF")
    except Exception as e:
        st.error(f"Backend connection error: {e}")


# =========================================================
# QUESTION GENERATION LOGIC
# =========================================================
def chatbot_ui():
    load_dotenv()
    st.set_page_config(page_title="Exam Questions Generator", page_icon="📘")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("📘 Exam Question Generator")
        st.caption(
            "Upload files and/or paste website URLs. You can use delimiters (; , || or new line). "
            "Optional labels: Unit1|https://example.com/page"
        )
    with col2:
        st.write(f"**User:** {st.session_state.user['username']}")
        if st.button("Logout"):
            logout()

    vectorstore, embedding_model = get_vector_store()

    uploaded_files = st.file_uploader(
        "Upload file(s)",
        accept_multiple_files=True,
        type=["pdf", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "docx", "doc", "txt"],
    )

    website_urls_text = st.text_area(
        "Website / Webpage URLs (optional) — use newline / ; / , / ||. Optional label: label|url",
        placeholder="Unit1|https://example.com/article ; Unit2|https://example.com/another-page\nhttps://example.com/page3 || https://example.com/page4",
    )

    website_items = parse_website_inputs(website_urls_text)

    question_type = st.selectbox(
        "Select question type",
        [
            "Objective",
            "Subjective",
            "Mathematical",
            "Both (Objective + Subjective)",
            "Coding Based",
        ],
    )

    num_objective = 0
    num_subjective = 0

    if question_type == "Objective":
        num_objective = st.number_input(
            "Number of objective questions", min_value=0, max_value=200, value=10, step=1
        )

    elif question_type == "Subjective":
        num_subjective = st.number_input(
            "Number of subjective questions", min_value=0, max_value=200, value=5, step=1
        )

    elif question_type == "Both (Objective + Subjective)":
        num_objective = st.number_input(
            "Number of objective questions", min_value=0, max_value=200, value=10, step=1
        )
        num_subjective = st.number_input(
            "Number of subjective questions", min_value=0, max_value=200, value=5, step=1
        )

    submit = st.button("Generate", disabled=(not uploaded_files and not website_items))

    if not uploaded_files and not website_items:
        st.info("Upload at least one file or enter at least one website URL to enable Generate.")

    prompt = PromptTemplate(
        template="""
You are an expert exam question and answer generator.

STRICT MODE:
Generate ONLY the question type selected below. Do not include other types.

SELECTED TYPE: {qtype}

COUNTS:
- Objective count: {obj_n}
- Subjective count: {sub_n}

SOURCES (use ONLY these; do not add outside knowledge):
{sources_block}

OUTPUT FORMAT (must follow exactly):
1) Questions
- Every question MUST end with: (Source: Sx)
2) Answer Key
- Every answer MUST end with: (Source: Sx)
- Sx must match the question’s source.

RULES:
- If SELECTED TYPE is "Objective":
  - Generate ONLY objective questions (MCQ / True-False / Fill in the blanks).
  - Do NOT generate subjective questions or long answers.
- If SELECTED TYPE is "Subjective":
  - Generate ONLY subjective questions (short/long descriptive).
  - Do NOT generate MCQs.
- If SELECTED TYPE is "Mathematical":
  - Generate ONLY numerical/problem-solving questions with final answers (+ steps only if present).
- If SELECTED TYPE is "Coding Based":
  - Generate ONLY coding/programming questions; include solutions only if present.
- If SELECTED TYPE is "Both (Objective + Subjective)":
  - Generate BOTH objective and subjective sections separately.

- Answers must come from the SOURCES only.
- If an answer is not found in the sources, write exactly: "Not found in PDF" (exact string).
""".strip(),
        input_variables=["qtype", "obj_n", "sub_n", "sources_block"],
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = StrOutputParser()
    chain = RunnableSequence(prompt, model, parser)

    if submit:
        sources = []
        image_files = []

        if website_items:
            with st.spinner("Fetching websites..."):
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futs = {ex.submit(cached_extract_website, url): (label, url) for (label, url) in website_items}
                    for fut in as_completed(futs):
                        label, url = futs[fut]
                        try:
                            web_text = fut.result()
                            sources.append(
                                {
                                    "name": (label or url),
                                    "type": "website",
                                    "meta": url,
                                    "text": web_text,
                                }
                            )
                        except Exception as e:
                            st.error(f"Could not fetch website content ({url}): {e}")

        def extract_file(f):
            file_bytes = f.read()
            name = (f.name or "")
            low = name.lower()

            if low.endswith(".pdf"):
                return {"name": name, "type": "file", "meta": name, "text": cached_extract_pdf(file_bytes)}
            elif low.endswith(".csv"):
                return {"name": name, "type": "file", "meta": name, "text": cached_extract_csv(file_bytes)}
            elif low.endswith(".xlsx") or low.endswith(".xls"):
                return {"name": name, "type": "file", "meta": name, "text": cached_extract_excel(file_bytes, name)}
            elif low.endswith(".docx"):
                return {"name": name, "type": "file", "meta": name, "text": cached_extract_docx(file_bytes)}
            elif low.endswith(".doc"):
                return {
                    "name": name,
                    "type": "file",
                    "meta": name,
                    "text": f"FILE: {name}\nLegacy .doc detected. Convert to .docx.\nNot found in PDF",
                }
            elif low.endswith(".txt"):
                return {"name": name, "type": "file", "meta": name, "text": cached_extract_txt(file_bytes, name)}
            elif low.endswith((".png", ".jpg", ".jpeg")):
                return {
                    "name": name,
                    "type": "image",
                    "meta": name,
                    "text": f"IMAGE FILE: {name}\nImage uploaded; OCR not enabled.\nNot found in PDF",
                    "image": (name, file_bytes),
                }
            else:
                return {"name": name, "type": "file", "meta": name, "text": f"FILE: {name}\nUnsupported.\nNot found in PDF"}

        if uploaded_files:
            with st.spinner("Extracting files..."):
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futs = [ex.submit(extract_file, f) for f in uploaded_files]
                    for fut in as_completed(futs):
                        try:
                            item = fut.result()
                            if item.get("type") == "image" and item.get("image"):
                                image_files.append(item["image"])
                                sources.append(
                                    {"name": item["name"], "type": "file", "meta": item["meta"], "text": item["text"]}
                                )
                            else:
                                sources.append(item)
                        except Exception as e:
                            sources.append(
                                {"name": "Unknown", "type": "file", "meta": "Unknown", "text": f"Extraction error: {e}\nNot found in PDF"}
                            )

        sources = [s for s in sources if s.get("text", "").strip()]
        if not sources:
            st.error("Could not extract any usable text from the inputs.")
            st.stop()

        for i, s in enumerate(sources, start=1):
            s["id"] = f"S{i}"

        sources_block = build_sources_block(sources, max_chars_per_source=20000)

        with st.spinner("Generating questions..."):
            result = chain.invoke(
                {
                    "qtype": question_type,
                    "obj_n": int(num_objective),
                    "sub_n": int(num_subjective),
                    "sources_block": sources_block,
                }
            )

        st.success("Done!")
        st.markdown(result)

        pdf_bytes = make_pdf_bytes("Exam Questions Generator Output", result)

        st.session_state.generated_pdf_bytes = pdf_bytes
        st.session_state.generated_result = result
        st.session_state.generated_question_type = question_type

        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_bytes,
            file_name="exam_questions.pdf",
            mime="application/pdf",
        )

        if st.button("Add to Database"):
            save_pdf_to_backend()

        if image_files:
            st.info("Images were uploaded. OCR is not enabled in this version.")
            with st.expander("Preview uploaded images"):
                for img_name, img_bytes in image_files:
                    st.image(img_bytes, caption=img_name, use_container_width=True)

        with st.expander("Show sources index (S1, S2, ...)"):
            for s in sources:
                st.write(f"**{s['id']}** — {s['name']} ({s['type']})")
                if s.get("type") == "website":
                    st.write(s.get("meta", ""))

    if st.session_state.generated_pdf_bytes:
        st.subheader("Previously Generated Output")
        st.download_button(
            label="⬇️ Download Last Generated PDF",
            data=st.session_state.generated_pdf_bytes,
            file_name="exam_questions.pdf",
            mime="application/pdf",
            key="download_last_pdf"
        )

        if st.button("Save Last Generated PDF to Backend", key="save_last_pdf"):
            save_pdf_to_backend()


# =========================================================
# MAIN APP FLOW
# =========================================================
def main():
    if not st.session_state.logged_in:
        st.set_page_config(page_title="Q&A Generator Login", page_icon="🔐")
        st.title("🔐 Q&A Generator Login")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            login_ui()

        with tab2:
            signup_ui()
    else:
        chatbot_ui()


if __name__ == "__main__":
    main()