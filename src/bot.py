import os
import time
import pathlib
import asyncio
import discord
from discord.ext import commands

from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

class MyBot(commands.Bot):
    _watcher: asyncio.Task

    def __init__(self, ext_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_dir = pathlib.Path(ext_dir)

    async def _load_extensions(self):
        print("Loading extensions...")
        print(f"Extension directory: {self.ext_dir}")
        for file in self.ext_dir.rglob("*.py"):
            print(f"Found file: {file}")
            if file.stem.startswith("_"):
                continue
            # Adjust the module path to account for the src directory
            relative_path = file.relative_to(self.ext_dir.parent)
            module_path = ".".join(relative_path.with_suffix("").parts)
            print(f"Attempting to load module: {module_path}")
            try:
                await self.load_extension(module_path)
                print(f"Loaded {file}")
            except commands.ExtensionError as e:
                print(f"Failed to load {file}: {e}")

    async def setup_hook(self):
        await self._load_extensions()
        self._watcher = self.loop.create_task(self._cog_watcher())

    async def _cog_watcher(self):
        print("Watching for changes...")
        last_mtimes = {}
        while True:
            await asyncio.sleep(60)  # Check for changes every second
            for file in self.ext_dir.rglob("*.py"):
                if file.stem.startswith("_"):
                    continue
                mtime = file.stat().st_mtime
                if file in last_mtimes:
                    if mtime != last_mtimes[file]:
                        print(f"Detected change in {file}, reloading...")
                        relative_path = file.relative_to(self.ext_dir.parent)
                        module_path = ".".join(relative_path.with_suffix("").parts)
                        try:
                            await self.unload_extension(module_path)
                            await self.load_extension(module_path)
                            print(f"Reloaded {file}")
                        except commands.ExtensionError as e:
                            print(f"Failed to reload {file}: {e}")
                last_mtimes[file] = mtime

# Initialize the bot with the src/cogs directory
bot = MyBot(ext_dir="./src/cogs", command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

bot.run(TOKEN)