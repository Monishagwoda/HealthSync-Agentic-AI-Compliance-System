import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import truststore
truststore.inject_into_ssl()
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Import agents
from agents.hipaa_agent import check_hipaa
from agents.icd_agent import check_icd
from agents.billing_agent import check_billing
from agents.summary_agent import final_summary

# Load ICD data
from utils.icd_loader import load_icd_data

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load ICD dataset
ICD_DATA = load_icd_data("C:/Users/MONISS/Downloads/archive/bs4_l4_dump/bs4_l4_dump")

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="HealthSync AI - Compliance Audit Agent",
    page_icon="🩺",
    layout="wide",
)

# -------------------------------------------------------------
# Custom CSS for Professional Look
# -------------------------------------------------------------
st.markdown("""
    <style>
        /* General page styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            margin: auto;
        }

        /* Title styling */
        h1 {
            text-align: center;
            color: #0B5394;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem;
        }

        /* Subtitle */
        h2, h3 {
            color: #1155CC;
            margin-top: 2rem;
        }

        /* Upload section */
        .upload-section {
            border: 2px dashed #0B5394;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            background-color: #F3F6FB;
        }

        /* Text areas */
        textarea {
            border-radius: 10px !important;
            border: 1px solid #ccc !important;
            font-size: 0.95rem !important;
        }

        /* Success message */
        .stSuccess {
            background-color: #E5F5E0 !important;
            border-left: 6px solid #3CB371 !important;
        }

        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
st.title("HealthSync AI")
st.markdown("<h4 style='text-align:center;color:gray;'>AI-Powered Medical Compliance & Audit System</h4>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------------
# File Upload Section
# -------------------------------------------------------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📄 Upload a Medical Document (PDF)", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner("Processing your document... Please wait ⏳"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        all_text = "\n".join([doc.page_content for doc in documents])

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(split_docs, embedding_model)
        retriever = vector_store.as_retriever()

    st.success("✅ PDF successfully uploaded and processed.")
    st.markdown("---")

    # ---------------------------------------------------------
    # Three-Column Layout for Reports
    # ---------------------------------------------------------
    with st.spinner("Running compliance audits using AI..."):
        hipaa_report = check_hipaa(all_text, retriever)
        icd_report = check_icd(all_text, retriever, ICD_DATA)
        billing_report = check_billing(all_text, retriever)
        summary_report = final_summary(hipaa_report, icd_report, billing_report)

    st.success("✅ Audit completed successfully.")
    st.markdown("---")

    # ---------------------------------------------------------
    # Results Display
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 HIPAA Compliance",
        "🧬 ICD Validation",
        "💳 Billing Analysis",
        "📊 Final Summary"
    ])

    with tab1:
        st.subheader("HIPAA Compliance Report")
        st.info("AI-based validation of data privacy, security, and protected health information handling.")
        st.text_area("HIPAA Report", hipaa_report, height=250)

    with tab2:
        st.subheader("ICD Code Validation Report")
        st.info("Verifies diagnosis codes and checks consistency with medical records.")
        st.text_area("ICD Report", icd_report, height=250)

    with tab3:
        st.subheader("Billing and Coding Analysis")
        st.info("Analyzes billing data for inconsistencies or potential upcoding.")
        st.text_area("Billing Report", billing_report, height=250)

    with tab4:
        st.subheader("Final Compliance Summary")
        st.success("Aggregated insights and compliance recommendations.")
        st.text_area("Summary", summary_report, height=300)

    # ---------------------------------------------------------
    # Footer
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:gray;'>© 2025 HealthSync AI | Powered by Gemini & LangChain</p>",
        unsafe_allow_html=True,
    )
