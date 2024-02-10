import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import Llama2
from llama_index import SimpleDirectoryReader

st.set_page_config(
    page_title="Chat with the PetCare docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Paw-sitive Pet Care: Expert Guidance for Happy Tails 🐶🐱")

if "messages" not in st.session_state.keys():  
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome!🐾 Paws and Reflect... Ask me a question about Basic PetCare!🐶"}
    ]


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome!🐾 Paws and Reflect... Ask me a question about Basic PetCare!🐶"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

st.sidebar.subheader("Models and Parameters")
selected_model = st.sidebar.selectbox(
    "Choose a model",
    ["Llama2-7B", "Llama2-13B", "Llama2-70B"],
    index=0,
)

temperature = st.sidebar.slider("Temperature", min_value=0.01, max_value=5.0, value=0.5, step=0.01)

llm_models = {
    "Llama2-7B": "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
    "Llama2-13B": "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    "Llama2-70B": "replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48"
}

llm_model = llm_models[selected_model]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Unleashing the Power of PetCare Docs! Fetching Knowledge... This may take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=Llama2(model=llm_model, temperature=0.5))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): 
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.text_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.empty():
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking..."):
        response = st.session_state.chat_engine.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.response})
