from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

billing_prompt = PromptTemplate(
    input_variables=["context"],
    template="""You are a healthcare billing auditor. Analyze this document: {context}
Determine if the diagnoses and treatments match the billing entries. Identify:
- Overbilling or underbilling
- Services billed but not documented
- Possible insurance fraud
List billing risks and provide an audit summary."""
)

def check_billing(text, retriever):
    chain = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/gemini-2.5-flash",  # Use full Gemini model name here
        temperature=0.1),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": billing_prompt}
    )
    return chain.run(text)
