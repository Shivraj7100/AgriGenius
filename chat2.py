from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 
from langchain_google_genai import ChatGoogleGenerativeAI 

# 2. Initialize Gemini using your Gemini API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=512,
    google_api_key="AIzaSyAQgu8LQRqtP2JFLCaU4hQGp6T6VTbHHpY" # Paste your Gemini API key here
)

# Set up the retrieval QA chain
def setup_retrieval_qa(db):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.6}
    )

    # Define your prompt template
    prompt_template = """You are Agri-Advisor, an expert agricultural assistant helping farmers make informed decisions about crop health, soil management, pest control, irrigation, and general farming practices.
    
     CONTEXT: {context}
     QUESTION: {question}"""

    PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])


    # 3. Everything else stays exactly the same!
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )
    return chain