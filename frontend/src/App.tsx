import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Bot,
  ChevronDown,
  Cpu,
  FolderOpen,
  Menu,
  MessageSquare,
  Plus,
  Send,
  Shield,
  X,
} from "lucide-react";

type Conversation = {
  id: string;
  project_id: string;
  title: string;
  updated_at: string;
};
type Message = { id: string; role: "user" | "assistant"; content: string };
type Capabilities = {
  inference_available: boolean;
  model_available: boolean;
  model_path: string;
  resources: { cpu_count: number; memory_gib: number | null; context_tokens: number };
};

function apiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (configured) return configured.replace(/\/$/, "");
  const host = window.location.hostname.replace(/-5173(?=\.app\.github\.dev$)/, "-8000");
  return `${window.location.protocol}//${host}:8000`;
}

const BASE = apiBase();
const storedToken = () => sessionStorage.getItem("nexus-token") ?? "";

export default function App() {
  const [project, setProject] = useState("default");
  const [projectDraft, setProjectDraft] = useState("default");
  const [token, setToken] = useState(storedToken);
  const [authRequired, setAuthRequired] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const headers = useMemo(
    () => ({
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    }),
    [token],
  );

  const request = useCallback(
    async (path: string, init?: RequestInit) => {
      const response = await fetch(`${BASE}${path}`, {
        ...init,
        headers: { ...headers, ...init?.headers },
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail ?? `Request failed (${response.status})`);
      }
      return response;
    },
    [headers],
  );

  const loadCapabilities = useCallback(async () => {
    try {
      const response = await request("/api/capabilities");
      setCapabilities(await response.json());
    } catch {
      setCapabilities(null);
    }
  }, [request]);

  const loadConversationList = useCallback(async () => {
    try {
      const response = await request(
        `/api/conversations?project_id=${encodeURIComponent(project)}`,
      );
      setConversations(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load conversations.");
    }
  }, [project, request]);

  const loadMessages = useCallback(
    async (id: string) => {
      try {
        const response = await request(
          `/api/conversations/${id}/messages?project_id=${encodeURIComponent(project)}`,
        );
        setMessages(await response.json());
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Could not load messages.");
      }
    },
    [project, request],
  );

  useEffect(() => {
    let active = true;
    fetch(`${BASE}/api/auth/status`)
      .then((response) => response.json())
      .then((status) => {
        if (active) setAuthRequired(status.required);
      })
      .catch(() => {
        if (active) setError(`Cannot reach the NEXUS API at ${BASE}. Start the API and refresh.`);
      });
    void loadCapabilities();
    return () => {
      active = false;
    };
  }, [loadCapabilities]);

  useEffect(() => {
    setConversationId("");
    setMessages([]);
    void loadConversationList();
  }, [project, loadConversationList]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const createConversation = async (
    clearMessages = true,
    title = "New conversation",
  ) => {
    setError("");
    try {
      const response = await request("/api/conversations", {
        method: "POST",
        body: JSON.stringify({ project_id: project, title: title.slice(0, 120) }),
      });
      const created: Conversation = await response.json();
      setConversationId(created.id);
      if (clearMessages) setMessages([]);
      setConversations((old) => [created, ...old]);
      setSidebarOpen(false);
      return created.id;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create conversation.");
      return "";
    }
  };

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    const text = draft.trim();
    if (!text || loading) return;
    setDraft("");
    setError("");
    setLoading(true);
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: text };
    const assistantId = crypto.randomUUID();
    setMessages((old) => [...old, userMessage, { id: assistantId, role: "assistant", content: "" }]);
    try {
      let activeConversation = conversationId;
      if (!activeConversation) {
        activeConversation = await createConversation(false, text);
        if (!activeConversation) throw new Error("Could not start a conversation.");
      }
      const response = await request(
        `/api/conversations/${activeConversation}/messages/stream`,
        { method: "POST", body: JSON.stringify({ project_id: project, content: text }) },
      );
      if (!response.body) throw new Error("Streaming is unavailable in this browser.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamError = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const data = JSON.parse(line.slice(6));
          if (data.type === "token") {
            setMessages((old) =>
              old.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + data.token }
                  : message,
              ),
            );
          } else if (data.type === "error") streamError = data.message;
        }
      }
      if (streamError) throw new Error(streamError);
      await Promise.all([loadConversationList(), loadCapabilities()]);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "The request failed.";
      setMessages((old) => old.filter((item) => item.id !== assistantId));
      setError(message);
      setDraft(text);
    } finally {
      setLoading(false);
    }
  };

  const chooseProject = (event: FormEvent) => {
    event.preventDefault();
    const next = projectDraft.trim();
    if (/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(next)) {
      setProject(next);
      setSidebarOpen(false);
      setError("");
    } else {
      setError("Project IDs must be 1-64 letters, numbers, dots, underscores, or hyphens.");
    }
  };

  return (
    <main className="app-shell">
      {sidebarOpen && <button className="mobile-backdrop" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand"><div className="brand-mark"><Bot size={21} /></div><span>NEXUS<span className="brand-ai"> AI</span></span><button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close menu"><X size={18} /></button></div>
        <button className="new-chat" onClick={() => void createConversation()}><Plus size={17} /> New conversation <span>⌘ K</span></button>
        <form className="project-picker" onSubmit={chooseProject}>
          <label htmlFor="project-id"><FolderOpen size={15} /> Project</label>
          <div className="project-entry"><input id="project-id" value={projectDraft} onChange={(event) => setProjectDraft(event.target.value)} aria-label="Project ID" /><button aria-label="Switch project"><ChevronDown size={16} /></button></div>
        </form>
        <div className="history-title">RECENT CONVERSATIONS</div>
        <div className="conversation-list">
          {conversations.map((conversation) => <button key={conversation.id} className={`history-item ${conversation.id === conversationId ? "history-active" : ""}`} onClick={() => { setConversationId(conversation.id); setSidebarOpen(false); void loadMessages(conversation.id); }}><MessageSquare size={15} /><span>{conversation.title}</span></button>)}
          {!conversations.length && <div className="empty-history">Your conversations will appear here.</div>}
        </div>
        <div className="sidebar-bottom">
          <div className="local-badge"><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} /><span>{capabilities?.inference_available ? "Local model online" : "Model not running"}</span></div>
          <div className="sidebar-foot">Private by design <Shield size={13} /></div>
        </div>
      </aside>
      <section className="main-panel">
        <header className="topbar"><div className="topbar-left"><button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={19} /></button><div className="crumb">Workspace <span>/</span> <b>{project}</b></div></div><div className="system-status"><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} />{capabilities?.inference_available ? "Inference ready" : "Waiting for model"}</div></header>
        <div className={`content ${messages.length ? "chat-content" : ""}`}>
          {messages.length === 0 ? <div className="welcome">
            <div className="welcome-orbit"><div className="welcome-logo"><Bot size={29} /></div><span className="orbit-dot orbit-one" /><span className="orbit-dot orbit-two" /></div>
            <div className="eyebrow"><span /> YOUR PRIVATE AI WORKSPACE</div>
            <h1>Think clearly.<br /><span>Build anything.</span></h1>
            <p className="welcome-description">A capable AI, running close to your work.<br />Your conversations stay in this workspace.</p>
            <div className="suggestions"><button onClick={() => setDraft("Explain how this project works.")}> <span>01</span> Explain how this project works <span>↗</span></button><button onClick={() => setDraft("Help me think through an idea.")}><span>02</span> Help me think through an idea <span>↗</span></button><button onClick={() => setDraft("Write a Python script that...")}><span>03</span> Write a Python script that... <span>↗</span></button></div>
            <div className="capability-card"><div className="capability-icon"><Cpu size={17} /></div><div className="capability-copy"><b>{capabilities?.model_path ?? "Qwen 2.5 · 0.5B"}</b><span>{capabilities?.inference_available ? "Running locally via llama.cpp" : capabilities?.model_available ? "Model downloaded · server stopped" : "Waiting for setup · CPU inference"}</span></div><div className="capability-resource"><Activity size={14} /><span>{capabilities ? `${capabilities.resources.cpu_count} CPU · ${capabilities.resources.context_tokens} ctx` : "Detecting resources"}</span></div></div>
          </div> : <div className="messages">{messages.map((message) => <article key={message.id} className={`message message-${message.role}`}><div className="message-avatar">{message.role === "user" ? "Y" : <Bot size={18} />}</div><div className="message-body"><div className="message-label">{message.role === "user" ? "YOU" : "NEXUS"} <span>·</span> {message.role === "assistant" ? "LOCAL MODEL" : "NOW"}</div><div className="message-text">{message.content || (loading ? <span className="typing"><i /><i /><i /></span> : "")}</div></div></article>)}<div ref={bottomRef} /></div>}
          {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button></div>}
        </div>
        {authRequired && <div className="auth-row"><Shield size={15} /><label htmlFor="api-token">API token</label><input id="api-token" type="password" autoComplete="off" placeholder="Paste NEXUS_API_TOKEN" value={token} onChange={(event) => { setToken(event.target.value); sessionStorage.setItem("nexus-token", event.target.value); }} /></div>}
        <form className="composer-wrap" onSubmit={sendMessage}><div className="composer"><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message NEXUS..." rows={1} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(event); } }} aria-label="Message NEXUS" /><div className="composer-tools"><span><span className="composer-dot" /> {capabilities?.inference_available ? "Local inference ready" : "Inference offline"}</span><button type="submit" disabled={!draft.trim() || loading} aria-label="Send message"><Send size={17} /></button></div></div><div className="composer-note">NEXUS can make mistakes. Verify important information.</div></form>
        <footer className="statusbar"><span><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} /> {capabilities?.inference_available ? "Connected to local model" : "Local model unavailable"}</span><span>PROJECT <b>{project}</b></span></footer>
      </section>
    </main>
  );
}
