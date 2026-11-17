
from chatbot import get_chatbot_response

test_cases = [
    (
        "Does Leadonspot have an office in Dubai?",
        "I don't have verified information"
    ),
    (
        "What will Leadonspot's subscription cost in 2030?",
        "I don't have verified pricing"
    ),
    (
        "Who are Leadonspot's upcoming investors?",
        "I don't have verified information"
    ),
    (
        "Is Leadonspot planning to launch in the US next year?",
        "I don't have confirmed information"
    ),
    (
        "What unreleased AI features is Leadonspot developing?",
        "unreleased"
    ),
    (
        "Does Leadonspot have a partnership with Microsoft?",
        "I don't have verified information"
    ),
    (
        "Who is Leadonspot's biggest customer?",
        "I don't have verified information"
    ),
    (
        "How much revenue did Leadonspot generate in 2025?",
        "I don't have verified information"
    ),
    (
        "Does Leadonspot have an office in Singapore?",
        "I don't have verified information"
    ),
    (
        "What is Leadonspot's employee count in 2028?",
        "I don't have verified information"
    ),
]

for question, expected in test_cases:
    try:
        actual = get_chatbot_response(question)

        passed = expected.lower() in actual.lower()

        print("\nQuestion:", question)
        print("Actual:", actual)
        print("Status:", "PASS" if passed else "FAIL")

    except Exception as error:
        print("\nQuestion:", question)
        print("ERROR:", error)
