import streamlit as st
import faiss
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


# model_name = "emilyalsentzer/Bio_ClinicalBERT"
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
api_key = os.getenv("OPENAI_API_KEY")
embeddings =  HuggingFaceEmbeddings(model_name = 'distilbert-base-uncased')
tokenizer = embeddings.client.tokenizer
# tokenizer.pad_token = tokenizer.eos_token
#
faiss_index = FAISS.load_local("final_clinical_trials_index", embeddings =embeddings, allow_dangerous_deserialization=True)
trial_descriptions = np.load("final_trial_descriptions.npy", allow_pickle=True)



def create_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

st.title("Clinical Trial Matching for Patients")
patient_profile = st.text_area("Enter the patient profile:")

llm = OpenAI(openai_api_key=api_key, temperature=0.5)

if st.button("Find Trials"):
    if patient_profile:
        patient_embedding = create_embedding(patient_profile)
        closest_trials = faiss_index.similarity_search_by_vector(patient_embedding, k=5)
        top_5_trials = [trial_descriptions[trial.metadata["id"]] for trial in closest_trials]
        trial_list = "\n".join([f"Trial {i+1}: {desc}" for i, desc in enumerate(top_5_trials)])
        prompt = f"""
        Based on the following patient profile:
        {patient_profile}
        Here are 5 potential clinical trials:
        {trial_list}
        Please choose the most relevant clinical trial for this patient and explain why it is the best match.
        """
        llm_response = llm(prompt)
        st.write("LLM Recommended Trial:")
        st.write(llm_response)
        st.write("Top 5 matching clinical trials:")
        for i, desc in enumerate(top_5_trials):
            st.write(f"- Trial {i+1}: {desc}")
    else:
        st.write("Please enter a patient profile.")
