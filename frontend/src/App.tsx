import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Bot,
  ChevronDown,
  Cpu,
  FilePlus2,
  FolderOpen,
  Menu,
  MessageSquare,
  Paperclip,
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
type MessageAttachment = { filename: string; size_bytes: number };
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: MessageAttachment[];
};
type ProjectFile = { path: string; size_bytes: number; modified_at: string };
type FileProposal = {
  projectId: string;
  path: string;
  content: string;
  summary: string;
};
type Capabilities = {
  inference_available: boolean;
  model_available: boolean;
  model_path: string;
  resources: { cpu_count: number; memory_gib: number | null; context_tokens: number };
};

function apiBase() {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (configured) return configured.replace(/\/$/, "");
  return "";
}

const BASE = apiBase();
const storedToken = () => sessionStorage.getItem("nexus-token") ?? "";
const attachmentExtensions = new Set([
  ".c", ".cc", ".cfg", ".cpp", ".cs", ".css", ".csv", ".diff", ".go",
  ".h", ".hpp", ".html", ".ini", ".java", ".js", ".json", ".jsx", ".log",
  ".md", ".php", ".py", ".rb", ".rs", ".sh", ".sql", ".svg", ".toml",
  ".ts", ".tsx", ".txt", ".xml", ".yaml", ".yml",
]);
const maxAttachmentBytes = 32_768;
const maxTotalAttachmentBytes = 65_536;

export default function App() {
  const [project, setProject] = useState("default");
  const [projectDraft, setProjectDraft] = useState("default");
  const [token, setToken] = useState(storedToken);
  const [authRequired, setAuthRequired] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [fileMode, setFileMode] = useState(false);
  const [filePath, setFilePath] = useState("");
  const [fileProposal, setFileProposal] = useState<FileProposal | null>(null);
  const [savingFile, setSavingFile] = useState(false);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const attachmentInputRef = useRef<HTMLInputElement>(null);

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

  const loadProjectFiles = useCallback(async () => {
    try {
      const response = await request(`/api/projects/${encodeURIComponent(project)}/files`);
      setProjectFiles(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load project files.");
    }
  }, [project, request]);

  const loadMessages = useCallback(
    async (id: string) => {
      try {
        setAttachments([]);
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
        if (active) setError("Cannot reach the NEXUS API. Confirm scripts/start.sh is running, then refresh.");
      });
    void loadCapabilities();
    return () => {
      active = false;
    };
  }, [loadCapabilities]);

  useEffect(() => {
    setConversationId("");
    setMessages([]);
    setFileProposal(null);
    setAttachments([]);
    void loadConversationList();
    void loadProjectFiles();
  }, [project, loadConversationList, loadProjectFiles]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const createConversation = async (
    clearMessages = true,
    title = "New conversation",
  ) => {
    setError("");
    const response = await request("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ project_id: project, title: title.slice(0, 120) }),
    });
    const created: Conversation = await response.json();
    setConversationId(created.id);
    if (clearMessages) {
      setMessages([]);
      setFileProposal(null);
      setAttachments([]);
    }
    setConversations((old) => [created, ...old]);
    setSidebarOpen(false);
    return created.id;
  };

  const startNewConversation = async () => {
    try {
      await createConversation();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create conversation.");
    }
  };

  const sendMessage = async (event: FormEvent) => {
    event.preventDefault();
    const pendingFiles = [...attachments];
    const text = draft.trim() || (pendingFiles.length ? "Please review the attached file(s)." : "");
    if (!text || loading || fileProposal) return;
    if (fileMode && !filePath.trim()) {
      setError("Enter a relative path for the file you want NEXUS to draft.");
      return;
    }
    setDraft("");
    setError("");
    setLoading(true);
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      attachments: pendingFiles.map((file) => ({ filename: file.name, size_bytes: file.size })),
    };
    const assistantId = crypto.randomUUID();
    setMessages((old) => [...old, userMessage, { id: assistantId, role: "assistant", content: "" }]);
    let conversationCreated = false;
    try {
      let activeConversation = conversationId;
      if (!activeConversation) {
        activeConversation = await createConversation(
          false,
          draft.trim() || pendingFiles.map((file) => file.name).join(", "),
        );
        conversationCreated = true;
      }
      const requestAttachments = await Promise.all(
        pendingFiles.map(async (file) => ({ filename: file.name, content: await file.text() })),
      );
      const response = await request(
        `/api/conversations/${activeConversation}/messages/stream`,
        {
          method: "POST",
          body: JSON.stringify({
            project_id: project,
            content: text,
            attachments: requestAttachments,
            create_file: fileMode,
            ...(fileMode ? { file_path: filePath.trim() } : {}),
          }),
        },
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
          } else if (data.type === "file_proposal") {
            if (
              typeof data.path !== "string" ||
              typeof data.content !== "string" ||
              typeof data.summary !== "string"
            ) {
              throw new Error("The model returned an invalid file draft.");
            }
            setFileProposal({
              projectId: project,
              path: data.path,
              content: data.content,
              summary: data.summary,
            });
            setMessages((old) => old.filter((message) => message.id !== assistantId));
          } else if (data.type === "error") streamError = data.message;
        }
      }
      if (streamError) throw new Error(streamError);
      setAttachments([]);
      await Promise.all([loadConversationList(), loadCapabilities(), loadProjectFiles()]);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "The request failed.";
      setMessages((old) =>
        old.filter(
          (item) =>
            item.id !== assistantId && (conversationCreated || item.id !== userMessage.id),
        ),
      );
      setError(message);
      setDraft(text);
    } finally {
      setLoading(false);
    }
  };

  const addAttachments = (files: FileList | null) => {
    if (!files?.length) return;
    const selected = Array.from(files);
    const currentBytes = attachments.reduce((total, file) => total + file.size, 0);
    const incomingBytes = selected.reduce((total, file) => total + file.size, 0);
    const invalid = selected.find((file) => {
      const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      return (
        file.name.startsWith(".env") ||
        /\.(pem|key|p12|pfx|secret|credentials)$/i.test(file.name) ||
        !attachmentExtensions.has(extension)
      );
    });
    if (invalid) {
      setError(`${invalid.name} is not a supported text/code file. Secret and binary files are not allowed.`);
    } else if (attachments.length + selected.length > 3) {
      setError("Attach up to 3 files per message.");
    } else if (currentBytes + incomingBytes > maxTotalAttachmentBytes) {
      setError("Attachments together must be no larger than 64 KiB.");
    } else if (selected.some((file) => file.size > maxAttachmentBytes)) {
      setError("Each attachment must be no larger than 32 KiB.");
    } else {
      setAttachments((current) => [...current, ...selected]);
      setError("");
    }
    if (attachmentInputRef.current) attachmentInputRef.current.value = "";
  };

  const approveFileProposal = async () => {
    if (savingFile || !fileProposal || fileProposal.projectId !== project) return;
    try {
      setSavingFile(true);
      setError("");
      await request(`/api/projects/${encodeURIComponent(project)}/files`, {
        method: "POST",
        body: JSON.stringify({
          path: fileProposal.path,
          content: fileProposal.content,
        }),
      });
      setFileProposal(null);
      setFileMode(false);
      setFilePath("");
      await loadProjectFiles();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not create project file.");
    } finally {
      setSavingFile(false);
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
        <button className="new-chat" onClick={() => void startNewConversation()}><Plus size={17} /> New conversation <span>⌘ K</span></button>
        <form className="project-picker" onSubmit={chooseProject}>
          <label htmlFor="project-id"><FolderOpen size={15} /> Project</label>
          <div className="project-entry"><input id="project-id" value={projectDraft} onChange={(event) => setProjectDraft(event.target.value)} aria-label="Project ID" /><button aria-label="Switch project"><ChevronDown size={16} /></button></div>
        </form>
        <div className="history-title">RECENT CONVERSATIONS</div>
        <div className="conversation-list">
          {conversations.map((conversation) => <button key={conversation.id} className={`history-item ${conversation.id === conversationId ? "history-active" : ""}`} onClick={() => { setConversationId(conversation.id); setSidebarOpen(false); void loadMessages(conversation.id); }}><MessageSquare size={15} /><span>{conversation.title}</span></button>)}
          {!conversations.length && <div className="empty-history">Your conversations will appear here.</div>}
        </div>
        <div className="history-title">PROJECT FILES</div>
        <div className="project-files">
          {projectFiles.map((file) => (
            <div className="project-file" key={file.path} title={`${file.path} · ${file.size_bytes} bytes`}>
              <FilePlus2 size={13} />
              <span>{file.path}</span>
            </div>
          ))}
          {!projectFiles.length && <div className="empty-history">Files you approve will appear here.</div>}
        </div>
        <div className="sidebar-bottom">
          <div className="local-badge"><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} /><span>{!capabilities ? "API unavailable" : capabilities.inference_available ? "Local model online" : "Model not running"}</span></div>
          <div className="sidebar-foot">Private by design <Shield size={13} /></div>
        </div>
      </aside>
      <section className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={19} /></button>
            <nav className="conversation-tabs" aria-label="Recent conversation tabs">
              {conversations.slice(0, 4).map((conversation) => (
                <button
                  key={conversation.id}
                  type="button"
                  className={`conversation-tab ${conversation.id === conversationId ? "conversation-tab-active" : ""}`}
                  onClick={() => {
                    setConversationId(conversation.id);
                    void loadMessages(conversation.id);
                  }}
                  title={conversation.title}
                >
                  <span>{conversation.title}</span>
                </button>
              ))}
              <button type="button" className="tab-new" onClick={() => void startNewConversation()} aria-label="New conversation"><Plus size={16} /></button>
            </nav>
          </div>
          <div className="system-status"><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} />{!capabilities ? "API unavailable" : capabilities.inference_available ? "Inference ready" : "Waiting for model"}</div>
        </header>
        <div className={`content ${messages.length ? "chat-content" : ""}`}>
          {messages.length === 0 && !fileProposal ? <div className="welcome">
            <div className="welcome-orbit"><div className="welcome-logo"><Bot size={29} /></div><span className="orbit-dot orbit-one" /><span className="orbit-dot orbit-two" /></div>
            <div className="eyebrow"><span /> YOUR PRIVATE AI WORKSPACE</div>
            <h1>Think clearly.<br /><span>Build anything.</span></h1>
            <p className="welcome-description">A capable AI, running close to your work.<br />Your conversations stay in this workspace.</p>
            <div className="suggestions"><button onClick={() => setDraft("Explain how this project works.")}> <span>01</span> Explain how this project works <span>↗</span></button><button onClick={() => setDraft("Help me think through an idea.")}><span>02</span> Help me think through an idea <span>↗</span></button><button onClick={() => setDraft("Write a Python script that...")}><span>03</span> Write a Python script that... <span>↗</span></button></div>
            <div className="capability-card"><div className="capability-icon"><Cpu size={17} /></div><div className="capability-copy"><b>{capabilities?.model_path ?? "API unavailable"}</b><span>{!capabilities ? "Cannot connect to the NEXUS API" : capabilities.inference_available ? "Running locally via llama.cpp" : capabilities.model_available ? "Model downloaded · server stopped" : "Waiting for setup · CPU inference"}</span></div><div className="capability-resource"><Activity size={14} /><span>{capabilities ? `${capabilities.resources.cpu_count} CPU · ${capabilities.resources.context_tokens} ctx` : "Resources unavailable"}</span></div></div>
          </div> : (
            <div className="messages">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={`message message-${message.role}`}
                >
                  <div className="message-avatar">
                    {message.role === "user" ? "Y" : <Bot size={18} />}
                  </div>
                  <div className="message-body">
                    <div className="message-label">
                      {message.role === "user" ? "YOU" : "NEXUS"} <span>·</span>{" "}
                      {message.role === "assistant" ? "LOCAL MODEL" : "NOW"}
                    </div>
                    <div className="message-text">
                      {message.content ||
                        (loading ? (
                          <span className="typing">
                            <i />
                            <i />
                            <i />
                          </span>
                        ) : (
                          ""
                        ))}
                    </div>
                    {!!message.attachments?.length && (
                        <div className="message-attachments">
                          {message.attachments.map((attachment) => (
                            <span className="message-attachment" key={attachment.filename}>
                              <Paperclip size={12} />
                              {attachment.filename}
                            </span>
                          ))}
                        </div>
                    )}
                  </div>
                </article>
              ))}
              {fileProposal && (
                <section className="file-proposal" aria-label="Review generated file">
                  <div className="file-proposal-heading">
                    <FilePlus2 size={16} />
                    <div>
                      <b>Review file before creating</b>
                      <span>{fileProposal.summary}</span>
                    </div>
                  </div>
                  <div className="file-proposal-path">{fileProposal.path}</div>
                  <textarea
                    className="file-proposal-content"
                    value={fileProposal.content}
                    onChange={(event) =>
                      setFileProposal({ ...fileProposal, content: event.target.value })
                    }
                    aria-label="Review and edit proposed file contents"
                  />
                  <div className="file-proposal-actions">
                    <button
                      type="button"
                      className="file-cancel"
                      onClick={() => setFileProposal(null)}
                      disabled={savingFile}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="file-approve"
                      onClick={() => void approveFileProposal()}
                      disabled={savingFile}
                    >
                      {savingFile ? "Creating…" : "Create file"}
                    </button>
                  </div>
                  <p>
                    Nothing is written until you choose “Create file”. New files only;
                    existing files are never overwritten.
                  </p>
                </section>
              )}
              <div ref={bottomRef} />
            </div>
          )}
          {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")} aria-label="Dismiss error"><X size={16} /></button></div>}
        </div>
        {authRequired && <div className="auth-row"><Shield size={15} /><label htmlFor="api-token">API token</label><input id="api-token" type="password" autoComplete="off" placeholder="Paste NEXUS_API_TOKEN" value={token} onChange={(event) => { setToken(event.target.value); sessionStorage.setItem("nexus-token", event.target.value); }} /></div>}
        <form className="composer-wrap" onSubmit={sendMessage}>
          <div className="composer">
            <input
              ref={attachmentInputRef}
              className="attachment-input"
              type="file"
              multiple
              accept={[...attachmentExtensions].join(",")}
              onChange={(event) => addAttachments(event.target.files)}
              aria-label="Choose text or code files to attach"
            />
            {!!attachments.length && (
              <div className="pending-attachments" aria-label="Files to attach">
                {attachments.map((file, index) => (
                  <span className="pending-attachment" key={`${file.name}-${index}`}>
                    <Paperclip size={12} />
                    {file.name}
                    <button
                      type="button"
                      onClick={() => setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                      aria-label={`Remove ${file.name}`}
                      disabled={loading}
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            {fileMode && (
              <input
                className="file-path-input"
                value={filePath}
                onChange={(event) => setFilePath(event.target.value)}
                placeholder="Relative file path, e.g. src/main.py"
                aria-label="New project file path"
              />
            )}
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                fileMode ? "Describe the file NEXUS should draft..." : "Message NEXUS..."
              }
              rows={1}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage(event);
                }
              }}
              aria-label={fileMode ? "Describe a project file" : "Message NEXUS"}
            />
            <div className="composer-tools">
              <button
                type="button"
                className="attach-button"
                onClick={() => attachmentInputRef.current?.click()}
                disabled={loading || attachments.length >= 3}
                aria-label="Attach text or code files"
                title="Attach text/code files (up to 3, 64 KiB total)"
              >
                <Paperclip size={16} />
              </button>
              <span>
                <span className="composer-dot" />{" "}
                {!capabilities
                  ? "API offline"
                  : capabilities.inference_available
                    ? "Local inference ready"
                    : "Inference offline"}
              </span>
              <button
                type="button"
                className={`file-mode-button ${fileMode ? "file-mode-active" : ""}`}
                onClick={() => {
                  setFileMode(!fileMode);
                  setError("");
                }}
                aria-pressed={fileMode}
              >
                <FilePlus2 size={14} /> {fileMode ? "File mode" : "Create file"}
              </button>
              <button
                type="submit"
                disabled={!draft.trim() || loading || Boolean(fileProposal)}
                aria-label="Send message"
              >
                <Send size={17} />
              </button>
            </div>
          </div>
          <div className="composer-note">
            {fileProposal
              ? "Review or cancel the pending file draft before sending another message."
              : fileMode
                ? "NEXUS drafts the file; review and approve it before it is saved."
                : "NEXUS can make mistakes. Verify important information."}
          </div>
        </form>
        <footer className="statusbar"><span><span className={`status-dot ${capabilities?.inference_available ? "online" : ""}`} /> {capabilities?.inference_available ? "Connected to local model" : "Local model unavailable"}</span><span>PROJECT <b>{project}</b></span></footer>
      </section>
    </main>
  );
}
