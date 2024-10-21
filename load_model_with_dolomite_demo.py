import transformers
from dolomite_engine import hf_models

if __name__ == "__main__":
    hf_models = transformers.AutoModelForCausalLM.from_pretrained('/proj/checkpoints/shawntan/hf_models/stickbreaking')
