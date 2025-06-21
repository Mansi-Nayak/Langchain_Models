# Langchain_Models
langchain

## Project Overview

This project demonstrates various capabilities of Langchain models by working with both open-source and closed-source LLMs, embeddings, prompt engineering, and structured outputs.

---

## Contents

1. **LLMs** with closed-source models.  
2. **ChatModels** using closed-source providers:  
   - OpenAI  
   - Anthropic  
   - Google  

3. **ChatModels** using open-source models from Hugging Face:  
   - via API  
   - local deployment  

4. **Embedding Models**  
   - Query-based search  
   - Document embedding  
   - Local similarity search  

5. **Langchain Prompting Techniques**  
   - Static prompts  
   - Dynamic prompts  
   - Prompt generators  
   - Prompt templates (`prompt_template`, `chat_prompt_template`)  
   - Message formatting (`chatbot_messages`, `messages`)  

6. **Structured Output Demos in Langchain**  
   - `TypedDict` based output  
   - Structured output demo (`s/o_demo`)  
   - Annotated output (`s/o_annotated`)  
   - Pydantic-based models:  
     - `with_structured_output_pydantic`  
     - `json_schema`  
     - `with_structured_output_json`

7. **Langchain output parser**
   - String Output Parser
   - String Output Parser 1
   - Structured Output Parser
   - Json Output Parser
   - Pydantic Output Parser

8. **Langchain Chains**
   - Simple Chain
   - Sequential Chain
   - Parallel Chain
   - Conditional Chain

9. **Langchain Runnables**
   - Simple LLM App
   - PDF Reader
   - LLM Chain
   - Retrieval QA Chain
   - Langchain demo problem Google Colab
   - Langchain demo solution Google Colab
   - Runnable Sequence
   - Runnable Parallel
   - Runnable Passthrough
   - Runnable Lambda
   - Runnable Branch

10. **Langchain Document Loaders**
   - Text Loader (cricket.txt)
   - PDF Loader (dl-curriculum.pdf)
   - Directory Loader (books(*.pdf))
   - Web Base Loader (url)
   - CSV Loader (Social_Network_Ads.csv)
---

## Technologies Used

- Python 3.10+
- Langchain
- Hugging Face Transformers
- OpenAI API
- Pydantic

---

