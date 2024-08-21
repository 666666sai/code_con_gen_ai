import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key='355417cf-37b8-4569-bfe1-7e6c733b3205')
index_name = "assembler-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

index = pc.Index(index_name)
response = index.describe_index_stats()

if response.total_vector_count == 0:
    pdf_loader = PyPDFLoader("assembler manual asmp1024_pdf.pdf")
    documents = pdf_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vector_store.add_documents(chunks)
# model="gpt-4o"
llm_qa = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm_qa, retriever=vector_store.as_retriever(), chain_type="stuff")

llm_codex = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")

prompt_template = """
You have an assembler code snippet that performs a specific task.
Your goal is to convert this assembler code into Python code,
maintaining the functionality and logic of the original code.

Theoretical Explanation:
{explanation}

Please provide the Python code with comments explaining each part.

{agent_scratchpad}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["explanation", "agent_scratchpad"])


def generate_python_code(explanation):
    formatted_prompt = prompt.format_prompt(explanation=explanation, agent_scratchpad="").to_string()
    response = llm_codex.invoke(formatted_prompt)
    return response.content


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e07b5f;
        color: #050505;
        font-family: Arial, sans-serif;
    }
    .stTextArea, .stFileUploader, .stButton {
        background-color: #3d405b;
        color: #c9aa8b;
    }
    .stTextArea textarea, .stFileUploader input, .stButton button {
        background-color: #81b29a;
        color: #FFFFFF;
        border-color: #050505;
    }
    .stButton button {
        background-color: #88bef2;
        color: #255f63;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #458cd1;
        color: #f2cc8f;
    }
    .title {
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
    }
    .title {
        font-size: 2em;
        color: #1E90FF; /* Font color for the title */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #110d80;ont-family: Magneto, sans-serif;"
    "font-size: 4em;margin-bottom: -20px;'>Code Master</h1>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.header("Assembler QA")

    question = st.text_area("Question...", "Enter your query here.", height=300, key="question")

    if st.button("Get Explanation", key="get_explanation"):
        with st.spinner("Generating Theory..."):
            prompt = f"""Based on the information you have in Pinecone,
            please describe the variable, constants and the assigned values.
            Then please describe what the following code does.
            Question: {question}"""

            answer = qa_chain.invoke({"query": prompt})
            theory_text = answer["result"]

            st.write("**Theory:**", theory_text)

            answer_with_question = f"Requirement:\n{question}\n\nAnswer:\n{theory_text}"

            st.download_button(
                label="Download Explanation",
                data=answer_with_question,
                file_name="Explanation.txt",
                mime="text/plain"
            )

with col2:
    st.header("Assembly to Python Converter")

    uploaded_file = st.file_uploader("Choose a text file with assembler code theory", type="txt", key="uploaded_file")

    if uploaded_file is not None:
        theoretical_explanation = uploaded_file.read().decode("utf-8")
        st.text_area("Assembler Code Theory", theoretical_explanation, height=300, key="assembler_code_theory")

        if st.button("Generate Python Code", key="generate_python_code"):
            with st.spinner("Generating code..."):

                try:
                    generated_code = generate_python_code(theoretical_explanation)
                    st.text_area("Generated Python Code", generated_code, height=500, key="generated_python_code")

                    st.download_button("Download Python Code as .py", data=generated_code, file_name="generated_code.py")
                    st.download_button("Download Python Code as .txt", data=generated_code, file_name="generated_code.txt")
                except Exception as e:
                    st.error(f"Error: {e}")
