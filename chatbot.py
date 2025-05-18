# Import the necessary classes from the transformers library.
# AutoTokenizer is used to convert text to tokens and back.
# AutoModelForSeq2SeqLM loads a sequence-to-sequence language model.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Specify the name of the pre-trained model to use.
# "facebook/blenderbot-400M-distill" is a conversational AI model.
model_name = "facebook/blenderbot-400M-distill"

# Load the pre-trained model from Hugging Face.
# The first run downloads the model; later runs use the cached version.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the corresponding tokenizer for the model.
# The tokenizer converts text to tokens that the model can process.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize an empty list to keep track of the conversation history.
conversation_history = []

# history_string = "\n".join(conversation_history)
# Creates a single string from the conversation history.
# This is needed to provide context to the model for generating relevant responses.

# input_text = "hello, how are you doing?"
# Example user input to test the chatbot before integrating with live user input.

# inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
# print(inputs)
# Tokenizes the conversation history and user input, converting them into tensors for the model.
# Printing helps verify the tokenization process and input structure.

# tokenizer.pretrained_vocab_files_map
# Shows the mapping of vocabulary files used by the tokenizer.
# Useful for debugging or understanding which files are loaded.

# outputs = model.generate(**inputs)
# print(outputs)
# Generates a response from the model using the tokenized input.
# Printing the raw output helps inspect the model's output tokens.

# response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
# print(response)
# Decodes the model's output tokens into a readable string.
# Printing the response verifies the decoding process.

# conversation_history.append(input_text)
# conversation_history.append(response)
# print(conversation_history)
# Updates the conversation history with the latest user input and model response.
# Printing the history helps ensure the conversation is tracked correctly.

# Start an infinite loop to interact with the user.
while True:
    # Combine all previous conversation turns into a single string.
    # This provides context for the model to generate relevant responses.
    history_string = "\n".join(conversation_history)

    # Prompt the user for input and read their message.
    input_text = input("> ")

    # Tokenize both the conversation history and the new user input.
    # encode_plus prepares the input for the model, returning tensors.
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate a response from the model using the tokenized input.
    outputs = model.generate(**inputs)

    # Decode the model's output tokens back into a readable string.
    # skip_special_tokens=True removes tokens like <s> and </s>.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Print the model's response to the user.
    print(response)

    # Add the user's input and the model's response to the conversation history.
    # This ensures the next response considers the full conversation.
    conversation_history.append(input_text)
    conversation_history.append(response)