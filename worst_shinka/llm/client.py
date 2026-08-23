
import openai
from worst_shinka.cli.settings import get_openrouter_api_key
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
TIMEOUT = 60*20
def get_client_llm(
) -> openai.OpenAI:
    client = openai.OpenAI(
        api_key=get_openrouter_api_key(),
        base_url=OPENROUTER_API_BASE,
        timeout=TIMEOUT
    )

    return client

