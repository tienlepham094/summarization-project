# 1. Introduction 
Summarization tools is a technique that can summarize text and pdf file in English and Vietnamese using abstractive summarization.
With PDF (long context) using RAG technique. 
- **Load PDF**: Use a library PyPDF to load the PDF and extract text.
- **Split Text into Chunks:**
Chunking: Split the text into manageable chunks. Using recursive text splitter.
Embedding vector: Use a pre-trained sentence embedding model e5 (intfloat/multilingual-e5-base) to convert each chunk into a vector embedding.
- **Retrieve Relevant Chunks with KMeans:**
Select the top k clusters that are most relevant to the query or context. This can be done by calculating the similarity of the query embedding with the cluster centroids.
- **Summarize Each Chunk:**
Summarization: Use the VinaLLaMa model to summarize each of the selected chunks.
Concatenation: Concatenate the summaries of the selected chunks into a single cohesive summary.
### Finetune model:
- BART (facebook/bart-base): finetune with samsum dataset ([dataset link](Samsung/samsum)) using to summarize English conversation and text.
[link](https://drive.google.com/drive/folders/1YT3gEFyOAxQOH0mmj5HmnlS_b62Up1q5?usp=sharing)
Finetuning code: [code](./notebook/dialoguesum.ipynb)
![](./image/Screenshot%20from%202024-05-29%2018-51-00.png)
- [Vinallama-2.7b-chat ](vilm/vinallama-2.7b-chat): finetune with vietnews dataset [link](harouzie/vietnews)
![](./image/Screenshot%20from%202024-05-29%2018-58-21.png)
![](./image/Screenshot%20from%202024-05-29%2018-59-35.png)
# 2. Technology 
- **Finetuning and implement**: Transformers, PyTorch
- **RAG technique**: Langchain
- **Backend**: FastAPI 
- **GUI**: Streamlit
# 3. Installation 
Install with python virtual environment
```
$ python3 -m venv venv
$ source venv/bin/activate
$ python3 -m pip install -r requirements.txt
```
# 4. Usage
After install libraries fill the .env see example file [.env.example]()
**Run vinallama in GG colab**
Upload [file](./notebook/HostLlama2BehindAPI.ipynb) to GG colab to use free GPU to run model. Deploy model by FastAPI and host with public ip ngrok
**Run bash file** 
```
bash run.sh
```
**Demo app**
![](./image/Screenshot%20from%202024-05-29%2021-03-57.png)
