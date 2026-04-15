from google.generativeai import GenerativeModel

import google.generativeai as genai
import os

# Set your API key here (you can also use environment variables instead)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the model
model = GenerativeModel("gemini-2.5-flash-lite")

print("Welcome to Simple Gemini Chatbot! Type 'exit' to quit.\n")


while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = model.generate_content(user_input)
    print("Chatbot:", response.text)