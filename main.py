import streamlit as st
import os
import json 
import requests

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

def get_content(content: str):
    index = content.find("<|im_start|>assistant")
    return content[index+len("<|im_start|>assistant"): index + len(content)]

def main(ip= None):
    st.title("Summarize your content")
    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    with st.sidebar:
        # options box
        option = st.selectbox(
            "Choose model",
            ("BARTsum", "Vinallama"),
            index=None,
            placeholder="Select model...",
            label_visibility="visible"
            )
        # slider
        max_length_value = st.slider(
        "Select a range of max_length",
        100, 512, 256)
        st.write("Max length:", max_length_value)
        
        temperature_value = st.slider(
        "Select a range of Temperature",
        0.0, 1.0, 0.7)
        st.write("Temperature:", temperature_value)
            
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # process pdf
    if uploaded_file is not None and option is not None:
        with open(uploaded_file.name, mode='wb') as w:
            w.write(uploaded_file.getvalue())
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load_and_split()
        
        text = ""
        for page in pages:
            text += page.page_content
        text = text.replace('\t', ' ')
        if text!="" and option=="BARTsum":
            with st.chat_message("assistant"):
                response = requests.post("http://0.0.0.0:8080/generate/bart-chain", json={"query": text,
                                                                                    "temperature": temperature_value,
                                                                                    "max_length": max_length_value})
                content = json.loads(response.text)["response"]
                if type(content) == list:
                    content = content[0]
                st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
        elif text != "" and option == "Vinallama" and ip is not None:
            with st.chat_message("assistance"):
                response = requests.post(f"{ip}/generate", json = {"inputs": text})
                content = json.loads(response.text)["response"]
                if type(content) == list:
                    content = content[0]
                st.markdown(get_content(content=content))
            st.session_state.messages.append({"role": "assistant", "content": get_content(content=content)})
        elif text != "" and option == "Vinallama" and ip is None: 
            response = "Can't find vinallama api"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        
    # Accept user input
    if user_query := st.chat_input("Enter your document to summarize!"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        


    if user_query!=None and option=="BARTsum":
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = requests.post("http://0.0.0.0:8080/generate/bart", json={"query": user_query,
                                                                                "temperature": temperature_value,
                                                                                "max_length": max_length_value})
            content = json.loads(response.text)["response"]
            if type(content) == list:
                content = content[0]
            st.markdown(content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": content})
    if user_query != None and option == "Vinallama" and ip is not None:
        with st.chat_message("assistance"):
            response = requests.post(f"{ip}/generate", json = {"inputs": user_query})
            content = json.loads(response.text)["response"]
            if type(content) == list:
                content = content[0]
            st.markdown(get_content(content=content))
        st.session_state.messages.append({"role": "assistant", "content": get_content(content=content)})
    if user_query != None and option == "Vinallama" and ip is None: 
        response = "Can't find vinallama api"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    elif option == None and user_query:
        response = "Please choose model to summarize"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


    # reset chat
    def reset_conversation():
        st.session_state.conversation = []
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.button('Clear Chat', on_click=reset_conversation)

if __name__ == "__main__":
    load_dotenv()
    API_MODEL = os.getenv("API_MODEL")
    main(API_MODEL)