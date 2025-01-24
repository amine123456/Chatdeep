#Hello Chatdeep
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a chatbot instance
chatbot = ChatBot('MyChatBot')

# Train the chatbot on English corpus
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

# Start chatting
while True:
    user_input = input("You: ")
    response = chatbot.get_response(user_input)
    print(f"Bot: {response}")