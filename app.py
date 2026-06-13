import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
import shutil
import gc

#Set up the page 
st.set_page_config(page_title="Clinical Trial Assistant", page_icon=":pill:", layout="wide")
st.title("Clinical Trial Assistant")
st.markdown("Upload a clinical trial protocol in PDF format and ask questions about it!")
from langchain_community.document_loaders import PyPDFLoader

#Load the uploaded document
def load_uploaded_documents(Uploaded_file):
    """load the document from uploaded PDF files
       create a temperoray files to process PDFs
       Handles multiple files efficiently"""
    documents = []
    temp_dir = tempfile.mkdtemp()
    try: 
        for idx, uploaded_file in enumerate(Uploaded_file):
            try:
                #save the uploladed file to temporary location
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                #load the document using PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                #add source information to metadata
                for doc in docs:
                    doc.metadata['source_file'] = uploaded_file.name
                    documents.append(doc)
                    doc.metadata['file_index'] = idx
                st.info(f"successfully loaded {uploaded_file.name} : {len(docs)} pages.")
            except Exception as e:
                st.warning(f"Failed to load {uploaded_file.name}: {e}")
                continue
        return documents
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded files: {e}")
        return documents
    finally:
        #clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

#Step 3:Split the document into chunks
def split_documents(documents, chunk_size = 800, chunk_overlap =150):
    #means each piece of texxt has 800 characters and chunk_overlap means pieces overlap a bit so we don't lose contact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", " ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
#Step4: Creating 'Memory box'
def create_vector_db(documents):
    # create a memory box where AI stores all document chunks.
    # It converts text into numbers that AI understands optimized for multiple files.
    if not documents:
        st.warning("no documents to process!")
        return None

    st.success(
        f"loaded {len(documents)} document pages from "
        f"{len(set([d.metadata.get('source_file', 'unknown') for d in documents]))} files!"
    )
# split documents into chunks
    chunks = split_documents(documents)
    st.success(f"split into{len(chunks)} chunks!")
#initialize distribution
    file_distribution = {}
#show file distribution = {}
    for chunk in chunks:
        file_name = chunk.metadata.get('source_file' , 'unknown')
        file_distribution[file_name] = file_distribution.get(file_name, 0) + 1
    with st.expander(f"File: {file_name}- chunks distribution by file"):
        for fn, count in file_distribution.items():
            st.write(f"--**{fn}**: {count} chunks")

#Create embeddings (convert text into numbers)
#Using a smaller model for better performance with multiple files
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={}
)
#Store in vector database with batching for better memory management 
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="clinical_trials",
            persist_directory="./chroma_db"
        )
#clear memory
        gc.collect()
        st.success("vector database created! ready to answer the questions.")
        return vectordb
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        st.info("Try uploading fewer or smaller files for better performance.")
        return None

#Step5:Create the AI chain
def create_qa_chain(vectordb):
    llm = OllamaLLM(model="llama3")
    response = llm.invoke("What is bioinformatics?")
    print(response)     
    template = """You are a helpful clinical trial research assistant...
Context: {context}
Question: {question}
Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join([
            f"[source: {doc.metadata.get('source_file','unknown')}] "
            f"(page {doc.metadata.get('page','?')})\n{doc.page_content}"
            for doc in docs
        ])

    qa_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain, retriever
#Step6:-Main application interface
#sidebar with instructions and file upload
with st.sidebar:
    st.header("How to use:")
    st.write("1. Upload docs: Click the Upload button and select multiple clinical trial documents.")
    st.write("2. Process docs: Click the Process button to analyzethe uploaded documents.")
    st.write("3. Ask questions: Type your question in the input field and click 'Submit'.")
    """For example, Questions:
    what is the dosage of the investigational drug?
    what are the inclusion criteria?
    Summarize the phase 2 results"""
    st.divider()
    st.header("System information")
    if 'vectordb' in st.session_state:
        st.success("system is ready!")
        if 'uploaded_file_names' in st.session_state:
            st.info(f"'Loaded file' :{len(st.session_state.uploaded_file_names)} file loaded")
            with st.expander("Loaded files"):
                for fname in st.session_state.uploaded_file_names:
                    st.write(f" . {fname}")
    else:
        st.warning("Please upload PDFs to start the system.")
    st.divider()
#Add reset button
if st.button("Reset Application", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        gc.collect()
        st.experimental_rerun()
    
#Step 7: The main interface for uploading and asking questions
#Main area: file upload section
st.header("Upload Clinical Trial Protocols")
st.info("**tips**: You can upload multiple PDF files at once. The system will analyze all of them together to answer your questions.")
uploaded_files = st.file_uploader(
    "Choose PDF files (can select multiple)", 
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files containing clinical trial protocols to upload."
)
if uploaded_files:
    st.success(f"{len(uploaded_files)} files(s) uploaded successfully!")
#show uploaded file name with size
with st.expander("uploaded files details"):
    total_size = 0
    for file in uploaded_files:
        size_kb = file.size/1024
        total_size += size_kb
        st.write(f"**{file.name}** - {size_kb:.2f} KB")
    st.write(f"**Total size**: {total_size:.2f} KB")
    if total_size> 10000: #10MB limit for better performance
        st.warning("Total file size exceeds 10MB. Consider uploading fewer or smaller files for better performance.")
#Process button (Delete any previous message/info stored)
if st.button("Process Documents", type= "primary"):
    with st.spinner("Processing documents.... This may take a moment ... "):
        try:
            #Clear previous data
            if 'vectordb' in st.session_state:
                del st.session_state.vectordb
            if 'qa_chain' in st.session_state:
                del st.session_state.qa_chain
            if 'retriever' in st.session_state:
                del st.session_state.retriever
            gc.collect()
            #Load and process documents from uploaded files
            st.info("loaded PDF files...")
            documents = load_uploaded_documents(uploaded_files)
            if documents:
                #create vector databases
                st.info("Creating vector databases...")
                vectordb = create_vector_db(documents)
                if vectordb:
                    #Create QA chain
                    st.info("Initializing AI chain...")
                    qa_chain, retriever = create_qa_chain(vectordb)
                with st.spinner("Processing documents... This may take a moment..."):
                    if qa_chain and retriever:
                        #Store in session state for Later use
                        st.success("Documents processed successfully! You can now ask questions about the clinical trial protocols.")
                        st.session_state.vectordb = vectordb
                        st.session_state.qa_chain = qa_chain
                        st.session_state.retriever = retriever
                        st.session_state.uploaded_file_names = [file.name for file in uploaded_files]
                        st.balloons()
                    else:
                        st.error("Failed to process documents.")
                    
            else:
                    st.error("Failed to create vector database.")
        except Exception as e:
                st.error(f"An error occurred while processing the documents: {e}")
                st.info("Try uploading fewer or smaller files for better performance.")
                st.info("1. Make sure ollama is running: 'Ollama Server'")
                st.info("2. Make sure the model is installed: 'Ollama pull genna: 2b'")
                st.info("3. If file are too large, try uploading fewer files at once")
                st.info("4. restart Ollamaif it is hanging: 'Ollama Server'(in a new terminal)")
st.divider()
            
#Question input section

#Step 8: Asking questions and getting answers.
#Question answering sections
st.header("Ask questions about your documents")
if 'qa_chain' not in st.session_state:
    st.info("Please upload and process PDF document first")
else:
    #Question input
    question = st.text_input(
        "Enter your question:", 
        placeholder="e.g., what is the primary endpoiont of the trial? Compare result across studies.", 
        key= "Question_input"
    )
    col1, col2 = st.columns([1,5])
    with col1:
        ask_button = st.button("Get answer", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear chat", use_container_width=True):
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
                st.rerun()
    if ask_button and question:
        with st.spinner("....Thinking...."):
            try:
                #get answer from the AI
                answer = st.session_state.qa_chain.invoke(question)
                #store chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"question": question, "answer": answer})
                #display chat history
                with st.expander("View source documents"):
                    source_docs = st.session_state.retriever.invoke(question)
                    for i, doc in enumerate(source_docs):
                        source_file = doc.metadata.get('source_file', 'unknown')
                        page_num = doc.metadata.get('page', 'N/A')
                        st.markdown(f'**Source {i+1}: {source_file} (page {page_num})**')
                        st.text(doc.page_content[:500] + " " if len(doc.page_content) > 500 else doc.page_content)
                        st.divider()
            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
        #Display chat history
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.divider()
            st.subheader("Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history [:-1])):
                with st.expander(f"**Question:**{chat['question'] [:50]}..., expander=false"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")

#Footer
st.divider()
st.markdown("""
<div style='text_align: Centric; color:gray:>
            <small>built with langchain, Ollama and streamlit | Clinical Trial AI Assistant v2.1- multiple support(gemma: 2b</small>)
</div>
""", unsafe_allow_html=True)          