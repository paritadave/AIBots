import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
#from llama_index.llms import OpenAI
#import openai
from llama_index import SimpleDirectoryReader
#from io import StringIO
import pandas as pd

st.set_page_config(
    page_title="Chat with the PetCare docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
#openai.api_key = st.secrets.openai_api_key
st.title("Paw-sitive Pet Care: Expert Guidance for Happy Tails 🐶🐱")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome!🐾 Paws and Reflect... Ask me a question about Basic PetCare!🐶"}
    ]


# Display or clear chat messages
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome!🐾 Paws and Reflect... Ask me a question about Basic PetCare!🐶"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Sidebar for model selection
st.sidebar.subheader("Models and Parameters")
selected_model = st.sidebar.selectbox(
    "Choose a model",
    ["GPT-3.5 turbo", "Mistral-7B", "Llama2-7B", "Llama2-13B", "Llama2-70B"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

if selected_model == "GPT-3.5 turbo":
    llm = "gpt-3.5-turbo"
    top_p = st.sidebar.slider("Top P", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider("Max Length", min_value=64, max_value=4096, value=512, step=8)
else:
    if selected_model == "Mistral-7B":
        llm = "mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70"
    elif selected_model == "Llama2-13B":
        llm = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
    elif selected_model == "Llama2-7B":
        llm = "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea"
    else:
        llm = "replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48"

    top_p = st.sidebar.slider("Top P", min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider("Max Length", min_value=64, max_value=4096, value=512, step=8)


@st.cache_resource(show_spinner=False)

def load_data():
    #with st.spinner(text="Loading and indexing the PetCare docs – hang tight! This should take 1-2 minutes."):
    with st.spinner(text="Unleashing the Power of PetCare Docs! Fetching Knowledge... This may take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        #service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in basic pet care information. Your job is to provide accurate and helpful advice related to pet care. Keep your answers factual and based on general pet care knowledge."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
