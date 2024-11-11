import streamlit as st
from openai import OpenAI
import os
from data_processing import get_chroma_index_for_pdf, create_educational_vectordb
from chatbot import process_chat_message
from study_materials import generate_study_materials, generate_downloads

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = ["An Introduction to Language and Linguistics.pdf"]

def main():
    st.set_page_config(page_title="📑 NLP Learning Plattform", layout="wide")
    initialize_session_state()
    
    # Title and description
    st.title("📑 NLP Learning Plattform")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.subheader("Upload Additional Custom NLP Learning Materials (to remove uploaded material and revert back to default, please refresh the page)")
        pdf_files = st.file_uploader(
            label="Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Process uploaded files
        files, filenames = process_uploads(pdf_files)
        vectordb, flagged_files = create_educational_vectordb(files, filenames)
        st.session_state["vectordb"] = vectordb
        
        display_upload_status(flagged_files)
        
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chatbot", "📚 Study Material Generator"])
    
    with tab1:
        # Create a container for chat history
        chat_container = st.container()
        
        # Create a container for input at the bottom
        input_container = st.container()
        
        # Display chat history in the chat container
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input at the bottom
        with input_container:
            if prompt := st.chat_input("Ask anything about NLP:"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Update chat container with new user message
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Get RAG context
                    context = ""
                    if vectordb:
                        search_results = vectordb.similarity_search(prompt, k=3)
                        for result in search_results:
                            context += f"\n{result.page_content}\n[Source: {result.metadata.get('filename')}, Page: {result.metadata.get('page')}]\n"
                    
                    # Generate streaming response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        for response in client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are an educational AI assistant specializing in NLP. Base your responses on the provided context and cite sources when possible."},
                                {"role": "assistant", "content": f"Context from documents: {context}"},
                                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
                            ],
                            stream=True,
                        ):
                            if response.choices[0].delta.content is not None:
                                full_response += response.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    
    with tab2:
        # Study material generation interface
        topic = st.text_input("Enter the topic you want to create materials for:")
        if st.button("Generate Materials", type="primary"):
            if not topic:
                st.warning("Please enter a topic first.")
            else:
                with st.spinner("Generating materials... This might take up to 1 minute. Please be patient 😇"):
                    content = generate_study_materials(vectordb, topic, client)
                    if content:
                        generate_downloads(content)

def process_uploads(pdf_files):
    # Initialize with hardcoded document
    with open("An_Introduction_to_Language_and_Linguistics.pdf", "rb") as f:
        hardcoded_pdf_data = f.read()
    
    files = [hardcoded_pdf_data]
    filenames = ["An Introduction to Language and Linguistics.pdf"]
    
    if pdf_files:
        for file in pdf_files:
            files.append(file.getvalue())
            filenames.append(file.name)
            if file.name not in st.session_state.uploaded_filenames:
                st.session_state.uploaded_filenames.append(file.name)
    
    return files, filenames

def display_upload_status(flagged_files):
    if flagged_files:
        st.warning("The following files were flagged as non-NLP relevant:")
        for file in flagged_files:
            st.write(file)
    
    st.divider()
    st.subheader("📚 Current Learning Materials")
    for filename in st.session_state.uploaded_filenames:
        st.write(f"- {filename}")

if __name__ == "__main__":
    main()