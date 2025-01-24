from transformers import pipeline

# Load a pre-trained chatbot model (DialoGPT-small)
chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    truncation=True,  # Explicitly enable truncation
    pad_token_id=50256,  # Set pad_token_id to eos_token_id
    max_length=50,  # Limit the response length
    do_sample=True,  # Enable sampling for temperature and top_p
    temperature=0.7,  # Adjust randomness (lower = more deterministic)
    top_k=50,  # Use top-k sampling
    top_p=0.95  # Use nucleus sampling
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