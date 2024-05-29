import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import HTTPException
from utils import get_bart_predict, get_bart_chain_predict
import uvicorn
load_dotenv()

BART_MODEL = os.getenv("BART_MODEL")
BART_TOKENIZER = os.getenv("BART_TOKENIZER")
app = FastAPI()


class TextInput(BaseModel):
    query: str
    temperature: float
    max_length:int
    
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate/bart")
async def generate_bart_summarization(data :TextInput):
    try: 
        if len(data.query) < 10:
            return {
                "response": "Too short sentence to summarize"
            }
        else:
            return {
                "response": get_bart_predict(model_path=BART_MODEL, 
                                            tokenizer_path=BART_TOKENIZER, 
                                            query=data.query,
                                            temperature=data.temperature,
                                            max_length= data.max_length)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
        
@app.post("/generate/bart-chain")
async def generate_bart_chain_summarization(data: TextInput):
    try:
        if len(data.query) < 10:
            return {
                "response": "Too short sentence to summarize"
            }
        else: 
            return {
                "response": get_bart_chain_predict(model_path=BART_MODEL,
                                                tokenizer_path=BART_TOKENIZER,
                                                query=data.query,
                                                temperature=data.temperature,
                                                max_length=data.max_length)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
