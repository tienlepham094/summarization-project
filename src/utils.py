import torch 
import transformers
import numpy as np 
from typing import List
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain import PromptTemplate 
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import HuggingFacePipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_bart_predict(model_path:str, 
                     tokenizer_path:str,
                     query:str, 
                     temperature:float, 
                     max_length: int):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tokenizer(query, 
                     max_length = 1024, 
                     truncation = True,
                    padding = "max_length", 
                    return_tensors = "pt")
    with torch.inference_mode():
        outputs = model.generate(input_ids = inputs["input_ids"].to(device),
                             attention_mask = inputs["attention_mask"].to(device),
                             max_length = max_length,
                             temperature =  temperature,
                             length_penalty = 0.8, 
                             num_beams = 2)
        decoded_outputs = [tokenizer.decode(s, skip_special_tokens=True,
                                clean_up_tokenization_spaces=True) for s in outputs]
    return decoded_outputs
    
def get_retrieval(text: str):
    # chunk data
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], 
                                               chunk_size=1024, 
                                               chunk_overlap=250)
    # embedding
    docs = text_splitter.create_documents([text])
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    vectors = embedding_model.encode([x.page_content for x in docs])
    # using kmeans to get selected chunks 
    if len(vectors) <=3:
        num_clusters = len(vectors)
    else:
        num_clusters = len(vectors) - 2
    closest_indices = []
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42).fit(vectors)
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i],
                                axis = 1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)
    selected_docs = [docs[doc] for doc in selected_indices]
    return selected_docs

def get_bart_chain_predict(model_path:str, 
                     tokenizer_path:str,
                     query:str, 
                     temperature:float, 
                     max_length: int):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    summarization_pipeline = transformers.pipeline(
    model = model,
    task = "summarization",
    max_length = 1024,
    tokenizer = tokenizer,
    temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=256,  # max number of tokens to generate in the output
)
    template = '''{text}'''
    prompt = PromptTemplate(template = template, input_variables = ["text"])
    my_pipeline = HuggingFacePipeline(pipeline=summarization_pipeline)
    summarization_chain = load_summarize_chain(llm=my_pipeline,
                             chain_type="stuff",
                             prompt = prompt)
    selected_docs = get_retrieval(query)
    results= []


    for i, doc in enumerate(selected_docs):
        chunk_summary = summarization_chain.run([doc])
        results.append(chunk_summary)
    summaries = "\n".join(results)
    print(results)
    return summaries