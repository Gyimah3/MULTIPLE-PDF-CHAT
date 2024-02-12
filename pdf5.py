import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import textract

# Load API key from environment variables( create '.env' file in the project directory in vs code to store your API KEY, e.g GOOGLE_API_KEY=" ")
API_KEY =st.secrets["GOOGLE_API_KEY"] # it will be 'os.getenv["GOOGLE_API_KEY"]' in Vscode
genai.configure(api_key=API_KEY)




import tempfile

def get_pdf_text(docs):
    text = ""
    for doc in docs:
        file_type = doc.name.split('.')[-1].lower()
        if file_type == 'pdf':
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += (page.extract_text() or "")
        elif file_type in ['txt', 'doc', 'docx']:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_type) as tmpfile:
                tmpfile.write(doc.getvalue())
                tmpfile_path = tmpfile.name
            # Process the temporary file
            text += textract.process(tmpfile_path).decode('utf-8')
        else:
            st.error("Unsupported file type: " + file_type)
    return text


# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    
     The following task involves reviewing a document, summarizing its contents, and providing insights as accurately as possible. If the user start a chat with any form of greetings respond with " Hey: my task involves reviewing  document, summarizing its contents, and providing insights as accurately as possible" 
     If the answer to a question is not contained within the provided document, and As an experienced engineer, your task is to review the document provided and extract critical technical details... Perform any task the user will ask, either summarizing or reviewing the  documents.Your summary should include key engineering concepts, design considerations, technological innovations, and any data or metrics that offer insights into the engineering work. If any queries arise that are not addressed within the document,
     Be an expert and read the user's mind and return accurate answer if the user's question is not complete or there relation between the user's question and the answer from the document .Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store locally
    new_db = FAISS.load_local("faiss_index", embeddings)
    # Perform the similarity search
    docs = new_db.similarity_search(user_question)
    # Get the conversational chain
    chain = get_conversational_chain()
    # Generate the response
    try:
        response = chain({"input_documents": docs, "question": user_question},
                         return_only_outputs=True)
        output_text = response.get("output_text", None)
        print(f"Generated response: {output_text}")  # Debug log
        return output_text if output_text else "Sorry, I don't have an answer for that."
    except Exception as e:
        print(f"An error occurred: {e}")  # Debug log
        st.error(f"An error occurred: {e}")
        return "An error occurred while processing your question, construct your question well please!."


    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    #print(response)
    #st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini💁")

    # Sidebar for file upload
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your PDF, Text, or Word Files",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx"]
        )
        if st.button("Process Documents"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state['vector_store'] = vector_store
                    st.success("Documents processed and indexed.")

    # Initialize chat history in session state if not present
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    st.write("## Chat")
    for author, message in st.session_state['chat_history']:
        color = "blue" if author == "You" else "green"
        st.markdown(f"<span style='color: {color};'>{author}:</span> {message}", unsafe_allow_html=True)

    # Input for user question
    user_question = st.text_input("Your question:", key="input")

    # When 'Send' button is pressed
    if st.button('Send'):
        if user_question and 'vector_store' in st.session_state:
            st.session_state['chat_history'].append(("You", user_question))
            response = user_input(user_question, st.session_state['vector_store'])
            if response:  # Only append if there's a response
                st.session_state['chat_history'].append(("Chatbot", response))
            else:
                st.session_state['chat_history'].append(("Chatbot", "Sorry, I don't have an answer for that."))
            # Move the chat display to the top
            st.experimental_rerun()

if __name__ == "__main__":
    main()

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()
