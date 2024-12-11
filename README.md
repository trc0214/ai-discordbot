# AI Chat Bot

This project is an AI-powered chat bot for Discord, utilizing various AI models and pipelines to provide intelligent responses and conversational capabilities.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ai-chat-bot.git
    cd ai-chat-bot
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add the following variables:
    ```env
    DISCORD_TOKEN=your_discord_token
    AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    ```

## Usage

1. Run the bot:
    ```sh
    python src/bot.py
    ```

2. The bot will log in to Discord and start listening for messages in the specified channels.

## Configuration

- **Discord Token**: Set your Discord bot token in the `.env` file.
- **Azure OpenAI**: Configure your Azure OpenAI endpoint and API key in the `.env` file.
- **Gemini API**: Set your Gemini API key in the `.env` file.

## Requirements

The project dependencies are listed in the `requirements.txt` file:

```txt
discord.py==2.4.0
python-dotenv==1.0.1
google-generativeai==0.8.3
openai==1.57.0
haystack-ai==2.8.0
google-ai-haystack==3.0.2
datasets==3.1.0
mongodb-atlas-haystack