from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import time

# Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_llm():
    """
    Initialize Groq LLM with proper LangChain integration.
    """
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        return ChatGroq(
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            groq_api_key=api_key,
            max_tokens=2048
        )
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        raise


def call_chain_with_retry(chain, inputs, max_retries=MAX_RETRIES):
    """
    Call chain with retry logic for rate limit errors.
    """
    for attempt in range(max_retries):
        try:
            response = chain.invoke(inputs)
            return response
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if '429' in error_msg or 'quota' in error_msg or 'rate limit' in error_msg:
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"⚠️  Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(
                        "Groq API rate limit exceeded. Please try again later or consider:\n"
                        "1. Wait a few minutes and try again\n"
                        "2. Use a different API key\n"
                        "3. Check your usage at https://console.groq.com"
                    )
            else:
                # Non-rate-limit error, raise immediately
                raise
    
    raise Exception("Max retries exceeded")


def get_conversational_chain():
    """Create conversational QA chain with optimized prompt."""
    prompt_template = """
    You are an intelligent assistant. Answer the question based on the provided context.
    
    Instructions:
    - Provide detailed and accurate answers
    - Use only information from the context
    - If the answer is not in the context, clearly state: "The answer is not available in the provided context"
    - Do not make up or infer information
    - Structure your answer clearly
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    model = get_llm()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_quiz_chain():
    """Create quiz generation chain with optimized prompt."""
    quiz_prompt = """
    You are an expert Quiz Generator AI.
    
    Generate exactly {num_questions} multiple-choice questions (MCQs) based ONLY on the provided context.
    
    Requirements:
    - Each question should have 4 options (A, B, C, D)
    - Questions should test understanding of key concepts
    - Provide the correct answer for each question
    - Use clear and unambiguous language
    
    Output Format:
    Q1. [Question text]?
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
    Correct Answer: [Letter]
    
    Context:
    {context}
    
    Generate {num_questions} questions now:
    """
    
    model = get_llm()
    prompt = PromptTemplate(
        template=quiz_prompt,
        input_variables=["context", "num_questions"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_summary_chain():
    """Create document summarization chain with optimized prompt."""
    summary_prompt = """
    You are a professional summarizer.
    
    Create a comprehensive summary of the provided context.
    
    Requirements:
    - Capture all key points and main ideas
    - Use clear and concise language
    - Organize information logically
    - Maintain accuracy to the original content
    
    Context:
    {context}
    
    Additional Instructions: {question}
    
    Summary:
    """
    
    model = get_llm()
    prompt = PromptTemplate(
        template=summary_prompt,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def check_api_health():
    """
    Quick test to check if the API is accessible.
    """
    try:
        llm = get_llm()
        test_response = llm.invoke("Say 'OK' if you can read this.")
        return True, "API is working"
    except Exception as e:
        error_msg = str(e).lower()
        if '429' in error_msg or 'quota' in error_msg or 'rate' in error_msg:
            return False, "API rate limit exceeded. Please wait or use a different key."
        return False, f"API error: {str(e)}"