import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pdfplumber
import docx
import re
import pandas as pd
from io import StringIO

# ✅ GenAI Test Case Generator setup
st.set_page_config("GenAI Test Case Generator", layout="wide")

# --- Custom CSS for visual enhancements
st.markdown(
    """
    <style>
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
        font-size: 0.85rem !important;
        white-space: nowrap;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea, .stTextInput input {
        font-size: 0.9rem;
    }
    .stDownloadButton button {
        border-radius: 8px;
        font-size: 0.85rem;
        padding: 0.4rem 1rem;
    }
    .stButton button {
        font-weight: bold;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    .card h3 {
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 GenAI Test Case Generator")
st.markdown("Generate intelligent test cases using local LLMs across any domain.")

# --- Sidebar settings
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Local LLM", ["mistral:latest", "llama3.2:latest", "openhermes:latest"])
test_case_count = st.sidebar.number_input("Number of Test Cases", min_value=1, value=10)

# Custom layout for test types
test_types = []

check_col, radio_col = st.sidebar.columns([2, 1])

with check_col:
    st.markdown("**Test Nature**")
    if st.checkbox("Positive", True): test_types.append("positive")
    if st.checkbox("Negative"): test_types.append("negative")
    if st.checkbox("Boundary"): test_types.append("boundary")
    if st.checkbox("Edge Case"): test_types.append("edge case")

with radio_col:
    st.markdown("**Test Strategy**")
    test_strategy = st.radio("", ["Functional", "Regression", "Unit Testing"], index=0)
    test_types.append(test_strategy.lower())

# --- Load LLM
llm = OllamaLLM(model=model_name, temperature=0.7)

# --- Domain Detection (Scored)
def detect_domain(text):
    domain_keywords = {
        "lte": ["3gpp", "emm", "eps", "mme", "enodeb", "nas", "bearer"],
        "banking": ["transaction", "account", "loan", "interest", "kyc"],
        "ecommerce": ["cart", "checkout", "order", "payment", "sku"],
        "healthcare": ["patient", "ehr", "prescription", "appointment"]
    }
    scores = {domain: 0 for domain in domain_keywords}
    text_lower = text.lower()
    for domain, keywords in domain_keywords.items():
        scores[domain] = sum(kw in text_lower for kw in keywords)
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else "general"

# --- Prompt Template Generator

def get_prompt(text, count, types):
    domain = detect_domain(text)
    domain_hint = {
        "lte": "Use LTE/3GPP terminology (e.g., EMM, NAS, MME, bearer).",
        "banking": "Focus on transactions, validations, balances, and regulatory logic.",
        "ecommerce": "Include cart, order, payment, and product workflows.",
        "healthcare": "Consider patient data, records, prescriptions, and scheduling.",
        "general": "Generate generic, clear test cases applicable to common systems."
    }
    type_guidance = {
        "functional": "Focus on verifying the system behaves according to requirements.",
        "regression": "Ensure previous functionality still works after changes.",
        "unit": "Target small isolated components or functions in the system."
    }
    strategy_type = [t for t in types.split(", ") if t in type_guidance.keys()]
    guidance = type_guidance.get(strategy_type[0], "") if strategy_type else ""

    full_prompt = f"""
You are a senior QA engineer.
{text}

Generate {count} structured {types} test cases.
{domain_hint.get(domain, '')}
{guidance}

Each test case must include:
- Test Case ID
- Title
- Description
- Input
- Expected Output
- Type

Only return the test cases in plain readable text.
"""
    return full_prompt

# --- Build chain
prompt_template = PromptTemplate(input_variables=["text", "count", "types"], template="{text}")
chain = LLMChain(llm=llm, prompt=prompt_template)

# --- File Parser

def extract_text(file):
    file_type = file.name.split(".")[-1].lower()
    with open(file.name, "wb") as f:
        f.write(file.read())
    if file_type == "pdf":
        with pdfplumber.open(file.name) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file_type == "docx":
        doc = docx.Document(file.name)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8")

# --- Parse Text Output to CSV

def parse_test_cases_to_csv(text):
    entries = re.split(r"(?=Test Case ID:)\s*", text.strip())
    rows = []
    for entry in entries:
        if not entry.strip(): continue
        row = {
            "Test Case ID": re.search(r"Test Case ID:\s*(.*)", entry),
            "Title": re.search(r"Title:\s*(.*)", entry),
            "Description": re.search(r"Description:\s*(.*)", entry),
            "Input": re.search(r"Input:\s*(.*)", entry),
            "Expected Output": re.search(r"Expected Output:\s*(.*)", entry),
            "Type": re.search(r"Type:\s*(.*)", entry)
        }
        cleaned = {k: (v.group(1).strip() if v else "") for k, v in row.items()}
        rows.append(cleaned)
    return pd.DataFrame(rows)

# --- UI Layout
left, right = st.columns(2)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Input Requirements")
    uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
    text_input = ""
    if uploaded_file:
        text_input = extract_text(uploaded_file)
        st.success("Text extracted from uploaded file.")
    else:
        text_input = st.text_area("Or paste your user story here:", height=250)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Generated Test Cases")
    if text_input.strip():
        if st.button("Generate Test Cases"):
            with st.spinner("Generating test cases with your local LLM..."):
                domain = detect_domain(text_input)
                full_prompt = get_prompt(text_input, test_case_count, ", ".join(test_types))
                chain.prompt.template = "{text}"
                output = chain.run(text=full_prompt, count=test_case_count, types=", ".join(test_types))

                st.text_area("Result", value=output, height=500)
                st.download_button("📅 Download Test Cases (Text)", data=output, file_name="test_cases.txt")

                try:
                    df = parse_test_cases_to_csv(output)
                    csv = df.to_csv(index=False)
                    st.download_button("📅 Download Test Cases (CSV)", data=csv, file_name="test_cases.csv", mime="text/csv")
                except Exception as e:
                    st.warning("⚠️ Couldn't parse test cases into CSV format. Try reviewing the output format.")
    else:
        st.info("Please upload a file or paste a user story to begin.")
    st.markdown('</div>', unsafe_allow_html=True)
