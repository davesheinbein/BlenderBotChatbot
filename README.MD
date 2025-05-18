# Simple BlenderBot Chatbot

This project is a simple command-line chatbot using the [facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill) conversational AI model from Hugging Face's Transformers library.

## Setup

1. **Clone the repository** (if you haven't already):

   ```sh
   git clone https://github.com/davesheinbein/BlenderBotChatbot.git
   cd BlenderBotChatbot
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```sh
   python3 -m venv my_env
   source my_env/bin/activate
   ```

3. **Install dependencies**:

   ```sh
   pip install transformers torch
   ```

## Running the Chatbot

To start the chatbot, run:

```sh
python3 chatbot.py
```

The first time you run this, the model will be downloaded from Hugging Face and cached for future runs.

## Usage

- After starting, the chatbot will prompt you with `> `.
- Type your message and press Enter.
- The bot will respond based on the conversation history.
- Continue chatting as long as you like. Press `Ctrl+C` to exit.

## What to Expect

- The chatbot uses the BlenderBot model, which is designed for open-domain conversation.
- Each turn, your input and the bot's response are added to the conversation history, so the bot can maintain context.
- Example interaction:

  ```
  > Hello!
  Hi there! How can I help you today?
  > What's the weather like?
  I'm not able to check real-time weather, but I hope it's nice where you are!
  ```

## Notes

- The script requires an internet connection for the initial model download.
- For best results, use Python 3.7 or newer.

---

Enjoy chatting!
