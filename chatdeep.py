# Import the necessary libraries
from transformers import pipeline

# Load a pre-trained chatbot model (DialoGPT)
chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-small",
    truncation=True,
    pad_token_id=50256,
    max_length=50,  # Limit the response length
    temperature=0.7,  # Adjust randomness
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
        response = chatbot(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chat()