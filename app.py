# TCAgentApp - Agentic-style Streamlit Test Case Generator (LLM-Decides Counts)

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import pdfplumber
import docx
import re
import yaml

def get_local_ollama_models():
    return ["llama3.2:latest"]

def build_prompt(text, count, test_type, strategy):
    examples = {
        "Positive": """Example:
Test Case ID: TC_POS_001
Title: Successful login
Description: User logs in with correct username and password.
Input: username = \"admin\", password = \"admin123\"
Expected Output: User is redirected to the dashboard
Type: Positive""",
        "Negative": """Example:
Test Case ID: TC_NEG_001
Title: Login with empty password
Description: Attempt login with valid username but empty password field
Input: username = \"admin\", password = \"\"
Expected Output: Error message: \"Password is required\"
Type: Negative""",
        "Edge Case": """Example:
Test Case ID: TC_EDGE_001
Title: Upload oversized file
Description: Upload a file exceeding maximum size limit
Input: file_size = 51MB (limit is 50MB)
Expected Output: Error: \"File size exceeds limit\"
Type: Edge Case""",
        "Boundary": """Example:
Test Case ID: TC_BOUND_001
Title: Password at minimum length boundary
Description: Submit a password with the exact minimum allowed characters
Input: password = \"123456\" (minimum = 6)
Expected Output: Password accepted, account created
Type: Boundary""",
    }
    sample = examples.get(test_type, "")
    return f"""
You are a senior QA engineer.
{text}

Generate {count} {test_type} {strategy} test cases in the following plain text format:

Test Case ID: TC_001
Title: ...
Description: ...
Input: ...
Expected Output: ...
Type: {test_type}

Each test case must be complete with all fields.

Follow this example:
{sample}

Return ONLY test cases in the exact format. Do NOT include headings or extra text.
"""

def plan_test_case_distribution(llm, requirement_text, strategy):
    prompt = f"""
You are a senior QA strategist.

Analyze the following requirement for {strategy} testing:

{requirement_text[:3000]}

Estimate the number of test cases needed in total.
Break them down into the following categories:
- Positive
- Negative
- Edge Case
- Boundary

Return a JSON object like this:
{{
  "positive": 10,
  "negative": 4,
  "edge": 2,
  "boundary": 1
}}

ONLY return the JSON. Do not explain anything.
"""
    try:
        chain = PromptTemplate.from_template("{text}") | llm
        result = chain.invoke({"text": prompt})
        match = re.search(r"\{.*?\}", result, re.DOTALL)
        return yaml.safe_load(match.group(0)) if match else {}
    except Exception as e:
        st.error(f"Planning agent failed: {e}")
        return {}

@st.cache_data
def extract_text(file):
    file_type = file.name.split(".")[-1].lower()
    with open(file.name, "wb") as f:
        f.write(file.read())
    if file_type == "pdf":
        with pdfplumber.open(file.name) as pdf:
            return "\n".join([p.extract_text() or "" for p in pdf.pages])
    elif file_type == "docx":
        doc = docx.Document(file.name)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode("utf-8")

def is_valid_test_case(tc_block):
    required_fields = ["Title:", "Description:", "Input:", "Expected Output:"]
    return all(field in tc_block and tc_block.strip().split(field)[-1].strip() for field in required_fields)

def main():
    st.set_page_config("GenAI Test Case Generator", layout="wide")
    st.title("🤖 GenAI Test Case Generator (LLM-Powered Planner)")
    st.markdown("Autonomously generate structured test cases using LLMs.")

    st.sidebar.header("⚙️ Settings")
    strategy = st.sidebar.radio("Test Strategy", ["Functional", "Regression"], index=0)

    import os

    base_url = "http://localhost:11434"
    if os.environ.get("DOCKER") == "1":
        base_url = "http://host.docker.internal:11434"

    llm = OllamaLLM(model="llama3.2:latest", base_url=base_url, temperature=0.2)

    left, right = st.columns(2)
    with left:
        st.subheader("📄 Input Requirements")
        uploaded_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
        text_input = extract_text(uploaded_file) if uploaded_file else st.text_area("Or paste requirements here:", height=250)

    with right:
        st.subheader("✅ Generated Test Cases")
        if text_input.strip() and st.button("Generate Test Cases"):
            with st.spinner("Planning how many test cases to generate using LLM..."):
                plan = plan_test_case_distribution(llm, text_input, strategy)

                if not plan:
                    st.error("Test plan could not be generated.")
                    return

                all_outputs = []
                for test_type, count in plan.items():
                    prompt = build_prompt(text_input, count, test_type.capitalize(), strategy)
                    try:
                        chain = PromptTemplate.from_template("{text}") | llm
                        output = chain.invoke({"text": prompt}).strip()

                        test_blocks = [tb.strip() for tb in re.split(r"(?=Test Case ID:)", output) if tb.strip()]
                        valid_cases = [tb for tb in test_blocks if is_valid_test_case(tb)]

                        if valid_cases:
                            all_outputs.append(f"--- {test_type.upper()} ---\n" + "\n\n".join(valid_cases))
                        else:
                            st.warning(f"⚠️ No valid {test_type} test cases generated.")
                    except Exception as e:
                        st.warning(f"⚠️ Error while generating {test_type} test cases: {e}")

                if all_outputs:
                    full_text = "\n\n".join(all_outputs)
                    st.text_area("🧾 Generated Test Cases", value=full_text, height=600)
                    st.download_button("📥 Download", data=full_text, file_name="test_cases.txt")
                else:
                    st.error("No valid test cases were generated.")

if __name__ == "__main__":
    main()
