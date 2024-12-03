# LLM_Document_Summarization4

## Load Model

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

## Resources
- [Hugging Face LLM model](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M)
- [LangChain](https://python.langchain.com/docs/introduction/)