# General Health Chatbot using OpenAI

from openai import OpenAI

# Replace with your API key
client = OpenAI(api_key="YOUR_API_KEY")

# Safety Filter
def safety_check(user_query):

    dangerous_keywords = [
        "suicide",
        "kill myself",
        "overdose",
        "self harm",
        "prescription without doctor",
        "illegal drugs"
    ]

    query_lower = user_query.lower()

    for word in dangerous_keywords:
        if word in query_lower:
            return False

    return True


# Chatbot Function
def health_chatbot(user_query):

    if not safety_check(user_query):
        return (
            "I'm unable to help with potentially harmful medical requests. "
            "Please consult a healthcare professional or emergency services."
        )

    system_prompt = """
    You are a helpful medical assistant.

    Rules:
    - Give simple and easy-to-understand answers.
    - Provide general health information only.
    - Do not diagnose diseases.
    - Do not prescribe medications.
    - Encourage users to consult healthcare professionals for serious concerns.
    - Keep responses friendly and concise.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_query}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content

# Main Chat Loop
print("=== General Health Chatbot ===")
print("Type 'exit' to quit.\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    answer = health_chatbot(query)
    print("\nChatbot:", answer)
    print()