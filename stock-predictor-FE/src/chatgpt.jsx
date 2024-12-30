import React, { useState } from "react";
import axios from "axios";

const ChatGPTQuery = () => {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    setError("");
    setResponse("");

    try {
      const result = await axios.post("http://localhost:5000/ask-chatgpt", {
        question,
      });
      setResponse(result.data.response);
    } catch (err) {
      setError(
        err.response?.data?.error || "An error occurred while querying ChatGPT."
      );
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h2>Ask ChatGPT</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="question" style={{ display: "block", marginBottom: "5px" }}>
            Enter your question:
          </label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows="4"
            style={{ width: "100%", padding: "8px" }}
            required
          ></textarea>
        </div>
        <button type="submit" style={{ padding: "10px 15px", cursor: "pointer" ,color:'red' }}>
          Submit
        </button>
      </form>
      {response && (
        <div style={{ marginTop: "20px", padding: "10px", background: "#f0f0f0" }}>
          <h4>Response:</h4>
          <p>{response}</p>
        </div>
      )}
      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <h4>Error:</h4>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default ChatGPTQuery;
