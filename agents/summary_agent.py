from langchain_google_genai import GoogleGenerativeAI

def final_summary(hipaa, icd, billing):
    final_prompt = f"""
You are a compliance summarizer agent.
Based on the following three reports:

HIPAA Report:
{hipaa}

ICD Code Report:
{icd}

Billing Audit Report:
{billing}

Provide a unified compliance summary highlighting the top concerns and action items.
"""
    llm = GoogleGenerativeAI(model="models/gemini-2.5-flash",  # Use full Gemini model name here
        temperature=0.1)
    response = llm.predict(final_prompt)
    return response
