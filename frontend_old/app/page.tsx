
"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setAnswer(data.answer);
    } catch {
      setAnswer("Unable to connect to the chatbot. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: "650px", margin: "80px auto", padding: "20px" }}>
      <h1>Leadonspot AI Assistant</h1>
      <p>Ask me anything about Leadonspot and its products.</p>

      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        placeholder="Ask your question..."
        style={{ width: "75%", padding: "12px" }}
      />

      <button onClick={sendMessage} disabled={loading}>
        {loading ? "Thinking..." : "Send"}
      </button>

      {answer && (
        <div style={{ marginTop: "25px", padding: "15px", background: "#f1f5f9" }}>
          <strong>Assistant:</strong>
          <p>{answer}</p>
        </div>
      )}
    </main>
  );
}
