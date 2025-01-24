# Import the necessary libraries
from transformers import pipeline

# Load a pre-trained chatbot model (DialoGPT)
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Function to handle the chat
def chat():
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot(user_input)[0]['generated_text']
        print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chat()