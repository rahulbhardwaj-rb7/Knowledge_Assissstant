import os
from dotenv import load_dotenv

load_dotenv()

def correct_spelling(text):
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ No Google API key found, skipping spell check")
            return text
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        prompt = f"""Please correct any spelling errors in the following text. 
Rules:
1. Only fix obvious spelling mistakes
2. Do NOT change proper names (like person names, company names, etc.)
3. Preserve the original meaning and structure
4. If the text is already correct, return it exactly as-is
5. Only return the corrected text, no explanations

Text to check: "{text}"

Corrected text:"""
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            corrected = response.content.strip()
        else:
            corrected = str(response).strip()
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1]
        if corrected.startswith("'") and corrected.endswith("'"):
            corrected = corrected[1:-1]
        
        return corrected
        
    except Exception as e:
        return text
