import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub import login

load_dotenv()

MY_TOKEN = os.getenv('HF_KEY')
login(new_session = False,
     token = MY_TOKEN)

model_id="Tien094/BARTpho-sum"
snapshot_download(repo_id=model_id, 
                  local_dir="BARTpho-sum",
                  local_dir_use_symlinks=False, revision="main")