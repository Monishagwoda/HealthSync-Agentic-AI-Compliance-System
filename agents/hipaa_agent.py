from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

hipaa_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a HIPAA compliance auditor AI. Given this medical text: {context}
Scan it for any HIPAA violations such as:
- Leakage of Protected Health Information (PHI)
- Missing access controls
- Unauthorized sharing
List any violations with reasoning.
"""
)

def check_hipaa(text, retriever):
    # This class supports API key
    llm = GoogleGenerativeAI(
        model="models/gemini-2.5-flash",  # Use full Gemini model name here
        temperature=0.1
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": hipaa_prompt}
    )

    return chain.run(text)
