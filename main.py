import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


api_key = "AIzaSyDbUIH_v0oLZPMWPBDCoc5HX3SvlCFDhr4"

# Configure the API with your API key
google.generativeai.configure(api_key=api_key)


def text_ext(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def Chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def vec_DB(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



conversation_history = []

def convo():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, retrieve it from the llm .\n\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """

    # Initialize Google model for text generation
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Define prompt template for the conversation
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Load the question-answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Modify the main function to incorporate conversation context
# Modify the main function to include a mechanism for maintaining conversation history
def main():
    st.set_page_config(page_title="RAG CHATBOT", page_icon=":robot:")

    # Custom CSS for changing background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6; /* You can change the color code here */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize conversation history using session_state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # File uploader at the top
    st.title("Upload your PDF Files")
    pdf_docs = st.file_uploader("Choose PDF Files", accept_multiple_files=True)

    # Button to trigger processing
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = text_ext(pdf_docs)
            text_chunks = Chunks(raw_text)
            vec_DB(text_chunks)
            st.success("Done")

    # Set up text input for user's prompt and reply inside the sidebar
    st.sidebar.title("CHATBOT")
    user_prompt = st.sidebar.text_input("User Prompt")

    # Process user's prompt
    if user_prompt:
        # Initialize Google embeddings for text understanding
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load documents from the FAISS index
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_prompt)

        # Initialize the conversation chain
        if len(st.session_state.conversation_history) == 0:
            chain = convo()
        else:
            chain = st.session_state.conversation_history[-1]["chain"]  # Use the last chain as the conversation chain

        # Obtain response from the conversation chain
        response_dict = chain({"input_documents": docs, "question": user_prompt})

        # Extract the response text
        response_text = response_dict.get('output_text', '')

        # Update conversation history
        st.session_state.conversation_history.append({"user_prompt": user_prompt, "chain": chain, "reply": response_text})

        # Update the conversation chain with the new history
        chain = convo()  # Initialize a new chain with updated history

        # Retrieve context from previous conversations
        previous_context = "\n".join([conv["reply"] for conv in st.session_state.conversation_history])

        # Add previous context to the user's prompt
        user_prompt_with_context = f"{previous_context}\n{user_prompt}"

        # Obtain response from the updated conversation chain considering the previous context
        response_dict_with_context = chain({"input_documents": docs, "question": user_prompt_with_context})

        # Extract the response text with context
        response_text_with_context = response_dict_with_context.get('output_text', '')

        # Update conversation history with the response including context
        st.session_state.conversation_history[-1]["reply"] = response_text_with_context

       

    # Display conversation history sequentially
    for conv in st.session_state.conversation_history:
        if conv["user_prompt"]:
            st.text(f"You: {conv['user_prompt']}")
        if conv["reply"]:
            st.text_area("Chatbot:", value=conv["reply"], height=100, max_chars=None, key=None)



if __name__ == "__main__":
    main()
