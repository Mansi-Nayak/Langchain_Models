import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variable
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace model
model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # You can try others too
    temperature=0.3,
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token,
)

# JSON Schema
json_schema = """
{
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg", "neutral"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}
"""

# Review Text
review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, 
or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 
45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images 
even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x 
loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still 
comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Review by Nitish Singh
"""

# Prompt without markdown backticks
prompt = f"""
You are an AI that extracts structured data from product reviews.

Below is the review:
\"\"\"{review_text}\"\"\"

Using the following JSON Schema, extract the required fields from the review and return a valid JSON:

{json_schema}

Return only the JSON object.
"""

# Get model response
response = model.invoke(prompt)
print("Raw response:", response)

# Print the actual text response
if hasattr(response, "content"):
    print(response.content)
else:
    print(response)

print(model.invoke("Say hello"))
