import os
from dotenv import load_dotenv
from discord.ext import commands
import google.generativeai as genai

load_dotenv()
ai_chat_channel_id = 1314151629796151307
api_key = os.getenv("GEMINI_API_KEY")
model = 'gemini-1.5-flash'
ai_theme = """talk short"""

class GenAICog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.bot_name = bot.user.name
        self.allowed_channels = [ai_chat_channel_id]
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.channel.id not in self.allowed_channels:
            return
        
        prompt = f"""
            Desired format: None
            People names: {message.author.display_name}
            Specific topics: None
            General themes: {ai_theme}

            Message: {message.content}
        """
        ai_response = self.model.generate_content(prompt).text
        await message.reply(ai_response)

async def setup(bot):
    await bot.add_cog(GenAICog(bot))

if __name__ == "__main__":
    # just for testing
    message = "Hello, AI!"
    client = genai.GenerativeModel(model)
    ai_response = client.generate_content(message).text
    print(ai_response)


        
