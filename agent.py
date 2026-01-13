import time
import threading
import logging

from rag.vector_store import SimpleFAISSStore, ingest_folder
from rag.retrieval import RetrievalAgent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("agent")

# ----------------------
# Init RAG components
# ----------------------
STORE = SimpleFAISSStore()
RETRIEVER = RetrievalAgent(STORE)

# Load knowledge documents
ingest_folder(STORE, "data/kb")

# ----------------------
# Heuristics
# ----------------------
BACKCHANNEL = {"yeah","ok","okay","hmm","uh-huh","right","mm","ah"}
INTERRUPT = {"stop","wait","no","hold on","pause"}
KNOWLEDGE_HINT = {"what","who","when","where","why","how","explain","describe"}


def is_backchannel(text: str):
    return text.lower().strip() in BACKCHANNEL


def is_interrupt(text: str):
    t = text.lower()
    return any(w in t for w in INTERRUPT)


def is_knowledge_query(text: str):
    t = text.lower().strip()
    first = t.split()[0]
    return ("?" in t) or (first in KNOWLEDGE_HINT)


# ----------------------
# Cancelable RAG task
# ----------------------
class CancellableTask:
    def __init__(self):
        self.cancelled_flag = False

    def cancel(self):
        self.cancelled_flag = True

    def cancelled(self):
        return self.cancelled_flag

    def run(self, question, callback):
        chunks = RETRIEVER.retrieve(question)

        if self.cancelled():
            log.info("Cancelled before prompt stage")
            return

        prompt = RETRIEVER.build_prompt(question, chunks)
        log.info("Prompt:\n%s", prompt)

        # Simulate LLM generation
        for _ in range(5):
            if self.cancelled():
                log.info("Cancelled during generation")
                return
            time.sleep(0.2)

        callback("Here is the answer grounded in context.", chunks)


current_task = None
agent_speaking = False


# ----------------------
# Event Handling
# ----------------------
def on_agent_state_change(state):
    global agent_speaking
    agent_speaking = (state == "speaking")
    log.info(f"[STATE] Agent → {state}")


def on_user_vad_start():
    log.debug("VAD started (sound detected)")


def on_user_transcript(text: str, is_final: bool):
    global current_task, agent_speaking
    log.info(f"STT (final={is_final}): {text}")

    if not is_final:
        return

    # Agent speaking ---------------------------------
    if agent_speaking:

        if is_interrupt(text):
            log.info("Hard interrupt detected → stopping agent")
            if current_task:
                current_task.cancel()
            # stop speech with LiveKit SDK
            return

        if is_backchannel(text):
            log.info("Backchannel detected → ignore")
            return

        # Not backchannel → treat as interruption
        log.info("Non-backchannel speech → interrupt")
        if current_task:
            current_task.cancel()
        return

    # Agent silent ------------------------------------
    if is_knowledge_query(text):
        log.info("Knowledge query detected → running RAG")

        task = CancellableTask()
        current_task = task

        def on_done(answer, chunks):
            if task.cancelled():
                return
            log.info("RAG Answer: %s", answer)
            # speak via LiveKit

        threading.Thread(target=task.run, args=(text, on_done), daemon=True).start()
        return

    log.info("Normal user message")
    # normal LLM reply here
