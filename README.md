# Real-Time Conversational Agent with RAG and Intelligent Interruption Handling

A real-time voice conversational agent that combines state-aware interruption handling with retrieval augmented generation (RAG). The agent ignores passive acknowledgements during speech, responds immediately to explicit interruptions, and grounds knowledge-based answers in a local document store.

---

## Features

- Real-time interruption handling using VAD and speech-to-text signals  
- Differentiates backchannels (e.g., “yeah”, “ok”) from real interrupts (e.g., “stop”)  
- Retrieval augmented generation using vector embeddings and FAISS  
- Document-grounded responses with cancellable retrieval and generation  
- Designed for low-latency live voice interaction

---

## Project Structure

