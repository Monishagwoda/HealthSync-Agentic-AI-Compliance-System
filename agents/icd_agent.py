from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

icd_prompt = PromptTemplate(
    input_variables=["context"],
    template="""You are a medical coding expert. Read this clinical text: {context}
Extract any diagnosis or symptoms and suggest the correct ICD-10 codes. Also, check if any codes mentioned are invalid or mismatched."""
)

def validate_icd_codes(llm_output, icd_df):
    matches = []
    if 'Code' not in icd_df.columns or 'Description' not in icd_df.columns:
        return ["ICD dataset is missing expected columns."]
    for code in icd_df['Code'].dropna().unique():
        if code in llm_output:
            row = icd_df[icd_df['Code'] == code]
            if not row.empty:
                matches.append(f"{code}: {row['Description'].values[0]}")
    return matches or ["No valid ICD codes matched."]

def check_icd(text, retriever, icd_df):
    chain = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/gemini-2.5-flash",  # Use full Gemini model name here
        temperature=0.1),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": icd_prompt}
    )
    result = chain.run(text)
    validated = validate_icd_codes(result, icd_df)
    return result + "\n\nValidated Matches:\n" + "\n".join(validated)
