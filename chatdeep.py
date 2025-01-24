# Import the necessary libraries
from transformers import pipeline

# Load a pre-trained chatbot model (DialoGPT)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

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