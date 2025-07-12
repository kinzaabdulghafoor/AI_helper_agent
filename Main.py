from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel

# Load secret key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if key exists
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing. Please add it in your .env file.")

# Gemini as OpenAI client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# New Agent with updated name & instructions
agent = Agent(
    name="HelperAgent",
    instructions="You're a smart assistant that gives clear answers to any user's question."
)

# Task input – slightly updated
task_input = ".what is tiktok"

# Run the agent
response = Runner.run_sync(
    agent,
    input=task_input,
    run_config=config
)

# Output the result
print("🔍 AI Agent Response:")
print(response)
