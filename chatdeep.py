from transformers import pipeline, BlenderbotSmallTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-blenderbot"
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_path)

# Create a chatbot pipeline
chatbot = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id,  # Use the correct pad_token_id
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Function to handle the chat
def chat():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Generate a response using the chatbot
        response = chatbot(user_input)[0]['generated_text']
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chat()