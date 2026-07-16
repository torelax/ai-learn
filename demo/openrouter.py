from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="openrouter_key",
)

# First API call with reasoning
response = client.chat.completions.create(
  model="deepseek/deepseek-v4-flash",
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ],
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

print("Assistant's answer:", response.content)

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.content,
    "reasoning_details": response.reasoning_details  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call - model continues reasoning from where it left off
response2 = client.chat.completions.create(
  model="deepseek/deepseek-v4-flash",
  messages=messages,
  extra_body={"reasoning": {"enabled": True}}
)

# Extract the assistant message with reasoning_details
response2 = response2.choices[0].message
print("Assistant's answer:", response2.content)