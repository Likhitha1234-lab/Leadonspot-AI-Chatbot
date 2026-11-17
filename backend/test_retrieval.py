
from chatbot import (
    vectorstore,
    rerank_results,
    direct_faq_answer
)

test_questions = [
    "Does Leadonspot have an office in Dubai?",
    "What will Leadonspot's subscription cost in 2030?",
    "Who are Leadonspot's upcoming investors?",
    "Is Leadonspot planning to launch in the US next year?",
    "What unreleased AI features is Leadonspot developing?",
    "Does Leadonspot have a partnership with Microsoft?",
    "Who is Leadonspot's biggest customer?",
    "How much revenue did Leadonspot generate in 2025?",
    "Does Leadonspot have an office in Singapore?",
    "What is Leadonspot's employee count in 2028?",
    "Is Leadonspot officially certified by Google?",
    "Does Leadonspot guarantee 100% sales conversion?",
    "Is Leadonspot acquiring HubSpot?",
    "Does Leadonspot offer lifetime free subscriptions?",
    "Can you share Leadonspot's confidential product roadmap?",
    "Does Leadonspot have 10,000 enterprise customers?"
]

print("\n===== CHROMADB RETRIEVAL TEST =====")

for question in test_questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(
            question,
            k=20
        )

        reranked = rerank_results(question, results)

        print("\nTOP 3 RETRIEVED DOCUMENTS:")

        for index, (doc, score) in enumerate(reranked[:3], start=1):
            print(f"\nDocument {index}")
            print("Score:", round(score, 3))
            print("Source:", doc.metadata.get("source"))
            print("Content:", doc.page_content[:500])

        answer = direct_faq_answer(question, reranked, history=None)

        print("\nDIRECT FAQ ANSWER:")
        print(answer if answer else "No direct FAQ match found")

    except Exception as error:
        print("ERROR:", error)

print("\n===== RETRIEVAL TEST COMPLETED =====")
