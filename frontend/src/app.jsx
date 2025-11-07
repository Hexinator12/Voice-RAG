import React, { useRef, useState } from "react";
import ReactDOM from "react-dom/client";

function App() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    setTranscript(""); setAnswer("");
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
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const uploadAudio = async (blob) => {
    setLoading(true);
    const fd = new FormData();
    fd.append("file", blob, "recording.webm");
    try {
      const res = await fetch("http://localhost:8000/upload-audio", { method: "POST", body: fd });
      const j = await res.json();
      setTranscript(j.transcript || "");
      setAnswer(j.answer || "");
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
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, fontFamily: "sans-serif" }}>
      <h2>RAG Voice Chat — Local</h2>
      <div>
        <button onClick={recording ? stopRecording : startRecording}>
          {recording ? "Stop" : "Record (mic)"}
        </button>
        {loading && <span style={{ marginLeft: 10 }}>Processing...</span>}
      </div>
      <div style={{ marginTop: 18 }}>
        <strong>Transcript:</strong>
        <div style={{ whiteSpace: "pre-wrap", background: "#111", color: "#fff", padding: 10, borderRadius: 6, marginTop: 6 }}>
          {transcript || "—"}
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <strong>AI Answer:</strong>
        <div style={{ whiteSpace: "pre-wrap", background: "#eee", padding: 10, borderRadius: 6, marginTop: 6 }}>
          {answer || "—"}
        </div>
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
