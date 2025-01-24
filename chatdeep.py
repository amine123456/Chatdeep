from transformers import pipeline

# Load the fine-tuned model
chatbot = pipeline(
    "text-generation",
    model="./fine-tuned-blenderbot",  # Use the fine-tuned model
    truncation=True,
    pad_token_id=50256,
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