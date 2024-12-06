import os
from dotenv import load_dotenv
from discord.ext import commands
from openai import AzureOpenAI

load_dotenv()
ai_chat_channel_id = 1314151662507524127
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key=os.getenv("AZURE_OPENAI_API_KEY") 
api_version="2024-02-01"
model="gpt-35-turbo-16k"
ai_theme = """talk short"""

class AzureOpenAICog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.bot_name = bot.user.name
        self.allowed_channels = [ai_chat_channel_id]
        self.client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.channel.id not in self.allowed_channels:
            return
        
        prompt = [
            {
                "role": "system",
                "content": ai_theme
            },
            {
                "role": "user",
                "content": message.content
            }
        ]
        
        ai_response = self.client.chat.completions.create(model=model , messages=prompt).choices[0].message.content
        await message.reply(ai_response)

async def setup(bot):
    
    await bot.add_cog(AzureOpenAICog(bot))

if __name__ == "__main__":
    # just for testing
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
    client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
    ai_response = client.chat.completions.create(model=model , messages=prompt).choices[0].message.content
    print(ai_response)


        
