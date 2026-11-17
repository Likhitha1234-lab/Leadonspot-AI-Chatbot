
from chatbot import get_chatbot_response

questions = [
    "Does Leadonspot operate in Dubai?",
    "Is there a Dubai branch of Leadonspot?",
    "Where are Leadonspot's international offices?",
    "What might Leadonspot cost in 2030?",
    "Who is investing in Leadonspot in the future?",
    "Is Leadonspot expanding to America?",
    "Tell me about Leadonspot's secret AI features.",
    "Is Microsoft officially associated with Leadonspot?",
    "Who is the company's largest client?",
    "What was the company's turnover in 2025?"
]

for question in questions:
    print("\n" + "=" * 60)
    print("Question:", question)

    try:
        answer = get_chatbot_response(question)
        print("Answer:", answer)
    except Exception as error:
        print("Error:", error)
