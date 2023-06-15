import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from chatTemplate import css , bot_template , user_template

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def pattern_matching(user_input):
    pattern = r'\b[pP][dD][Ff]+s?\b'
    replaced_input = re.sub(pattern , "document" , user_input,flags=re.IGNORECASE)
    return replaced_input

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    chat_html = ""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            chat_html += user_template.replace("{{MSG}}", message.content) + "\n"
        else:
            chat_html += bot_template.replace("{{MSG}}", message.content) + "\n"

    # Update chat section with newest message
    st.markdown(chat_html, unsafe_allow_html=True)

    # Scroll to the bottom
    scroll_down_script = """
    <script>
    var chatElement = document.getElementsByClassName("chat-container")[0];
    chatElement.scrollTop = chatElement.scrollHeight;
    </script>
    """
    st.markdown(scroll_down_script, unsafe_allow_html=True)
    
    # Scroll to the top
    scroll_up_script = """
    <script>
    var chatElement = document.getElementsByClassName("chat-container")[0];
    chatElement.scrollTop = 0;
    </script>
    """
    st.markdown(scroll_up_script, unsafe_allow_html=True)

def get_conv_chain(vectorstore):
    llm = ChatOpenAI()
  

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def main():
    load_dotenv()
    st.set_page_config(page_title="DocuChat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.title("DocuChat :books:")

    col1, col2 = st.columns([4, 1])
    user_question = col1.text_input("Ask a question about your documents:")

    if col2.button("Ask"):
        with st.spinner("Processing your question..."):
            if user_question:
                handle_userinput(user_question)

    st.markdown("---")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        
        
        
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_text(pdf_docs)
                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conv_chain(vectorstore)

if __name__ == "__main__":
    main()
