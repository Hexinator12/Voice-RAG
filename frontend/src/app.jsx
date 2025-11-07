import React, { useRef, useState } from "react";
import ReactDOM from "react-dom/client";

function App() {
  const [messages, setMessages] = useState([]); // {role: 'user'|'assistant'|'system', text}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const appendMessage = (msg) => setMessages((m) => [...m, msg]);

  const sendText = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    appendMessage({ role: "user", text: q });
    setLoading(true);
    try {
      // Backend accepts JSON or form; use JSON
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q, k: 4 })
      });
      const j = await res.json();
      const answer = j.answer || "";
      appendMessage({ role: "assistant", text: answer || "(no answer)" });
    } catch (e) {
      appendMessage({ role: "system", text: "Request failed: " + e.message });
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    if (loading) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await uploadAudio(blob);
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      setRecording(true);
    } catch (e) {
      alert("Mic permission denied or unsupported browser.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const uploadAudio = async (blob) => {
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", blob, "recording.webm");
      const res = await fetch("http://localhost:8000/upload-audio", { method: "POST", body: fd });
      const j = await res.json();
      const transcript = j.transcript || "";
      const answer = j.answer || "";
      if (transcript) appendMessage({ role: "user", text: transcript });
      appendMessage({ role: "assistant", text: answer || "(no answer)" });
      if (j.audio_base64) {
        const binary = atob(j.audio_base64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
        const audioBlob = new Blob([bytes.buffer], { type: "audio/wav" });
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audio.play();
      }
    } catch (e) {
      appendMessage({ role: "system", text: "Audio upload failed: " + e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif" }}>
      <h2>Voice-RAG Chat</h2>
      <div style={{ margin: "12px 0", display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendText()}
          placeholder="Type your question..."
          style={{ flex: 1, padding: "10px 12px", borderRadius: 8, border: "1px solid #ccc" }}
        />
        <button onClick={sendText} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
        <button onClick={recording ? stopRecording : startRecording}>
          {recording ? "Stop" : "🎤 Record"}
        </button>
      </div>
      <div style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12, minHeight: 260 }}>
        {messages.length === 0 && (
          <div style={{ color: "#666" }}>Ask something or use the mic to speak.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>{m.role}</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);

