import os
from openai import AzureOpenAI, APIConnectionError
from dotenv import load_dotenv

def test():
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_version = "2024-02-01"

    # Print the environment variables to verify they are loaded correctly
    print(f"api key: {api_key}")
    print(f"endpoint: {azure_endpoint}")

    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )

    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo-16k",  # Use a supported model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                {"role": "user", "content": "Do other Azure AI services support this too?"}
            ]
        )

        print(response.choices[0].message.content)

    except APIConnectionError as e:
        print(f"Failed to connect to Azure OpenAI: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

print("before dotenv")
test()

# with dotenv
print("with dotenv")
load_dotenv()
test()