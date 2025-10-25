import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

## Now we used llm for prompt 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os

## Page Details 
st.set_page_config(page_title="Simple LangChain Chatbot with GROQ", page_icon="🪐")
st.title("Simple Lang Chain Chat With Groq")
st.markdown("Learn LangChain basics with Groq's ultra fast interface")

## side bar 
with st.sidebar:
    st.header("Setting")

    ## talk about API key 
    api_key = st.text_input("GROQ_API_KEY", type="password", help="Get free api key at console ")



    ## Model selection Drop Down 
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0
    )

    ## clear Button 
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

## Initilize Chat history 
if "messages" not in st.session_state:
    st.session_state.messages = []

## Initilize LLM 
@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    ## initlize the Groq Model 
    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=0.7,
        streaming=True
    )

    ## Create Prompt template 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer questions clearly."),
        ("user", "{question}")
    ])

    ## create a chain 
    chain = prompt | llm | StrOutputParser()
    return chain


chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.markdown("Get your free API key from [console.groq.com](https://console.groq.com)")

else:
    ## display message 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    ## chat input 
    if question := st.chat_input("Ask me anything"):
        ## add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        ## Generate Response 
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # stream response from GROQ 
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "|")

                message_placeholder.markdown(full_response)

                ## ADD to History 
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")

## Examples 
st.markdown("---")
st.markdown("### Try these examples ")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology?")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a note about AI.")

## Footer 
st.markdown("-----")
st.markdown("Built with LangChain & Groq")
st.markdown("Mubashir Yaqoob")
