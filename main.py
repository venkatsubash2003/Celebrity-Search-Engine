import os
from constants import openai_key
from langchain.llms import OpenAI 
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
os.environ["OPENAI_API_KEY"] = openai_key

st.title("Celebrity Search Engine")

input_text = st.text_input("Using LangChain")
llm = OpenAI(temperature=0.8)

first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about the person named {name}"
)

person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='dob',memory_key='chat_history')

chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)


second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born?"
)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "what major events occured during {dob} of that person around the world?"
)
chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=description_memory)

parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables = ['name'],output_variables=['person','dob','description'],verbose=True)






if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander("Person Name"):
        st.info(person_memory.buffer)
    with st.expander("Major Events"):
        st.info(description_memory.buffer)