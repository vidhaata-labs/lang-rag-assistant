# Lang RAG Assistant: UX flows

Source of truth for the screens: the approved Doop canvas [Lang RAG Assistant](https://doop.design/c/3XoSXpnr0Q), which has pages Overview, Chat, History, Documents, Model and Design system.

Every element in these flows is a native Streamlit component. The Streamlit command for each element is listed in section 11.

Items marked **Open point** are not in the wireframes or were not confirmed. They are collected in section 12 for approval.

---

## 1. Page map

The app has four pages in a sidebar navigation (`st.navigation` with `st.Page`). Chat is the default page.

```mermaid
flowchart LR
    Start([App opens]) --> Gate{Any provider key configured?}
    Gate -- No --> ChatBlocked[Chat page: no model configured]
    Gate -- Yes --> Chat[Chat page]
    ChatBlocked -- Open Model page --> Model[Model page]
    Chat <-->|sidebar| History[History page]
    Chat <-->|sidebar| Documents[Documents page]
    Chat <-->|sidebar| Model
    History <-->|sidebar| Documents
    Documents <-->|sidebar| Model
    History -- Resume --> Chat
    Documents -- Suggested question --> Chat
    Model -- Use this model --> Chat
```

---

## 2. First run and model gating

Doop frames: Chat · No model configured (`EYCJN-8k_B`), Model · Configure (`Ox0gxjGEvs`).

```mermaid
flowchart TD
    A[Load settings from .env] --> B{Any provider key found?}
    B -- No --> C[Chat: info alert 'No model configured']
    C --> D[Chat input disabled: 'Choose a model to start chatting']
    C --> E[Button: Open Model page]
    E --> M[Model page]
    B -- Yes --> F[Model options: models of configured providers only]
    F --> G[Default chat model and embedding model selected]
    G --> H[Chat input enabled]
```

---

## 3. Choose a model (Model page)

Doop frames: Model · Configure (`Ox0gxjGEvs`), Model · Provider without key (`FkOEHsTCFm`).

```mermaid
flowchart TD
    A[Model page] --> B[Select chat model: one model per provider, configured providers only]
    B --> D[Select embedding model, default all-MiniLM-L6-v2 local]
    D --> E[Provider key table shows Configured or Missing key]
    E --> F{Click Use this model}
    F -- Selection is valid --> G[Save selection]
    G --> H[Success confirmation]
    H --> I[Chat uses the selected model]
    F -- Saved provider has no key --> J[Warning alert: No API key for provider]
    J --> K[Use this model is disabled]
    K --> L[Add key to .env and restart, or pick another provider]
```

---

## 4. Ask a question and get a cited answer

Doop frames: Chat · Empty state (`rwjJNpYRhs`), Chat · Answer with sources (`h6nBr0rxtv`), Chat · Generating (`_CXI8sszwm`).

Starting points are typing into the chat input or clicking a suggested question. Both go through the same path.

```mermaid
sequenceDiagram
    actor U as User
    participant C as Chat page
    participant A as Agent
    participant I as Document index
    participant L as Model provider
    participant D as SQLite

    alt Starts from a suggestion
        U->>C: Click suggested question
    else Types a question
        U->>C: Type in chat input and press Enter
    end
    C->>D: Create or continue conversation
    C-->>U: Show user message
    C-->>U: Status: Searching documents
    C->>A: Send question with conversation history
    A->>I: Search indexed passages
    I-->>A: Matching passages
    C-->>U: Status: Reading N passages
    A->>L: Ask model with passages
    L-->>A: Answer text
    A-->>C: Answer with source list
    C->>D: Save assistant message and cited passages
    C-->>U: Assistant message with source line
    C-->>U: Cited passages expander (closed)
    C-->>U: Follow-up question pills
```

**Follow-up:** a follow-up question uses the same path. The conversation history is sent with it.

**Following a suggestion:** clicking a follow-up pill runs the same path as typing that question.

---

## 5. Stop a running answer

Doop frame: Chat · Generating (`_CXI8sszwm`). The chat input shows a Stop button while an answer is being produced.

```mermaid
flowchart TD
    A[Answer is being produced] --> B[Chat input shows Stop button]
    B --> C{User clicks Stop?}
    C -- No --> D[Answer completes]
    D --> E[Answer with sources, as in flow 4]
    C -- Yes --> F[Generation is cancelled]
    F --> G[Partial answer stays on screen, marked Stopped] --> H[Partial turn is saved]
    H --> I[Chat input is ready for the next question]
```

Step G is decided: the core streams answer tokens, so the text streamed so far is saved as the answer and marked Stopped.

---

## 6. Provider error while answering

Doop frame: Chat · Provider error (`10BpD4Db9k`). The question stays visible above the error.

```mermaid
flowchart TD
    A[Question sent] --> B{Provider responds?}
    B -- Yes --> C[Answer, as in flow 4]
    B -- No --> D{Error type}
    D -- Rate limit --> E[Alert: Rate limit reached]
    D -- Quota exhausted --> F[Alert: Quota exhausted]
    D -- Provider unavailable --> G[Alert: Provider unavailable]
    D -- Key rejected --> H[Alert: API key rejected]
    E --> I[Button: Try again]
    G --> I
    E --> J[Button: Change model]
    F --> J
    H --> J
    I --> A
    J --> K[Model page]
```

The wireframe shows only the rate-limit alert. Quota, unavailable and key-rejected messages are an **Open point**.

---

## 7. Resume a conversation (History)

Doop frames: History · Saved conversations (`pyrwXMg5_6`), History · Empty (`IUgPU07tjH`).

```mermaid
flowchart TD
    A[History page] --> B{Any saved conversations?}
    B -- No --> C[Info alert: No conversations yet]
    C --> D[Button: Go to Chat]
    D --> E[Chat page]
    B -- Yes --> F[List, newest first]
    F --> G[Search box filters the list]
    G --> F
    F --> H[Click Resume on a conversation]
    H --> I[Chat page shows the saved messages and sources]
    I --> J[User continues with a new question, as in flow 4]
```

---

## 8. Add a PDF (Documents)

Doop frames: Documents · Library (`2_tdH7XC9j`), Documents · Upload and indexing (`Q5lTDNl-YB`), Documents · Invalid PDF error (`h6UYB94Y-j`).

```mermaid
flowchart TD
    A[Documents library] --> B[Click Add a PDF]
    B --> C[Drop a PDF or click Browse files]
    C --> D{Can the file be read as a PDF?}
    D -- No --> E[Alert: Couldn't read this file]
    E --> F[Button: Remove file]
    E --> G[Button: Choose another PDF]
    F --> C
    G --> C
    D -- Yes --> H[File name and size shown]
    H --> I[Button: Add this PDF]
    I --> J[Status: Indexing the file]
    J --> K[Library shows the new file as Uploaded, with its suggested-question count]
```

**Note:** the library cards show the file's page count and its number of suggested questions. A new file's suggested questions appear once indexing finishes.

Indexing failure is an **Open point**: the wireframes do not show a failed indexing state.

---

## 9. Suggested questions (Documents to Chat)

Doop frame: Documents · Library (`2_tdH7XC9j`). Each document has a Suggested questions expander, which is collapsed in the wireframes.

```mermaid
flowchart LR
    A[Documents library] --> B[Expand Suggested questions for a document]
    B --> C[Click a question]
    C --> D[Chat page opens]
    D --> E[The question is sent, as in flow 4]
```

---

## 10. Navigation between pages

Doop frame: Overview · App map (`x6-e9WEWK_`).

```mermaid
flowchart LR
    Chat -- sidebar --> History
    Chat -- sidebar --> Documents
    Chat -- sidebar --> Model
    History -- sidebar --> Chat
    History -- sidebar --> Documents
    History -- sidebar --> Model
    Documents -- sidebar --> Chat
    Documents -- sidebar --> History
    Documents -- sidebar --> Model
    Model -- sidebar --> Chat
    Model -- sidebar --> History
    Model -- sidebar --> Documents
```

The active page is highlighted in the sidebar with the accent colour (section 14).

---

## 11. Screen inventory

| Page | State | Doop frame | Streamlit elements |
|------|-------|------------|--------------------|
| Chat | Empty state with suggestions | `rwjJNpYRhs` | `st.chat_input`, `st.pills`, `st.caption` |
| Chat | Answer with sources | `h6nBr0rxtv` | `st.chat_message`, `st.status` (compact), `st.expander`, `st.pills` |
| Chat | Generating with Stop | `_CXI8sszwm` | `st.chat_input(submit_mode="stop")`, `st.status` (steps) |
| Chat | Provider error | `10BpD4Db9k` | `st.chat_message`, `st.error(title=...)`, `st.button` ×2 |
| Chat | No model configured | `EYCJN-8k_B` | `st.info(title=...)`, `st.button`, `st.chat_input` (disabled), `st.sidebar.selectbox` (disabled) |
| History | Saved conversations | `pyrwXMg5_6` | `st.text_input`, `st.container(border=True)`, `st.button` (Resume) |
| History | Empty | `IUgPU07tjH` | `st.info(title=...)`, `st.button` |
| Documents | Library | `2_tdH7XC9j` | `st.button` (Add a PDF), `st.container(border=True)`, `st.badge`, `st.expander` |
| Documents | Upload and indexing | `Q5lTDNl-YB` | `st.file_uploader`, `st.button`, `st.status`, `st.progress` |
| Documents | Invalid PDF | `h6UYB94Y-j` | `st.file_uploader`, `st.error(title=...)`, `st.button` |
| Model | Configure | `Ox0gxjGEvs` | `st.selectbox` ×2, `st.button` (Use this model), `st.table` or `st.dataframe` for keys |
| Model | Provider without key | `FkOEHsTCFm` | `st.selectbox`, `st.warning(title=...)`, `st.button` (disabled) |

---

## 12. Open points

These need your approval before implementation. My recommendation is in each entry.

1. **Decided: the model picker lists only models whose provider has a configured key.** There is one model per provider, so the chat model picker is the only chooser and there is no separate provider picker. The "provider without key" frame covers only the case where a saved model loses its key.
2. **Decided: Inter for body and headings, JetBrains Mono for code.** Set in `.streamlit/config.toml` with Streamlit's Google Fonts URL syntax, so the app renders the same faces as the wireframes:

   ```toml
   font = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
   headingFont = "Inter:https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
   codeFont = "'JetBrains Mono':https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap"
   ```

3. **Stop keeps the partial answer,** marks it Stopped, and saves it. Recommended. The wireframes do not show the marker.
4. **Quota, unavailable and key-rejected messages** use the same alert pattern as rate limit, with the copy written during implementation. Recommended.
5. **Changing the embedding model** would need documents to be re-indexed. Not in the wireframes. Proposal: warn the user and re-index on confirm.
6. **Model selection is saved** to the local database, so it survives a restart. Not in the wireframes. Proposal: save it.
7. **History search** matches conversation titles only. Not in the wireframes. Proposal: titles only.
8. **Indexing failure** shows an error alert, and the file is not added to the library. Not in the wireframes. Proposal: add it, using the Invalid PDF pattern.
9. **Regenerate suggestions** was in the earlier app but is not in the wireframes, so it is excluded. Add it if wanted.

---

## 13. Start a new chat (added after approval)

Added at your request. The Chat sidebar has a "New chat" button, below the model picker. It leaves the current conversation, so the next question starts a new one and the empty state with its suggested questions returns. Saved conversations stay in History.

```mermaid
flowchart TD
    A[Chat page with a conversation open] --> B[Click New chat in the sidebar]
    B --> C[Current conversation is left, not deleted]
    C --> D[Empty state with suggested questions]
    D --> E[Next question starts a new conversation, as in flow 4]
    C --> F[History still lists the earlier conversation]
```

Decided after the review: Stop saves the partial answer marked Stopped (the core streams tokens); document cards show pages and six suggested questions, as in the wireframes; the upload limit is 200 MB, as in the wireframes.

---

## 14. Sidebar brand, navigation and footer (added after the build)

Added after the build, so the Doop frames do not show it yet. Every page uses the same sidebar, top to bottom:

1. **Brand row.** The app icon with the title "Lang RAG Assistant" beside it, as one image above the menu. Title in Inter Bold, 16px, colour `#1F1B16`, 10px gap from a 32px icon with 8px corner radius. The browser tab shows the icon alone.
2. **Menu.** Chat, History, Documents, Model. The active item uses the accent colour `#B45309`, semibold. No collapse or expand toggle, because the wireframes show none and the sidebar stays open.
3. **Chat only:** the chat model picker and the New chat button (section 13).
4. **Footer.** "Powered by Vidhaata Labs", 13px Inter in the muted token `#6B6158`, pinned to the bottom-left of the sidebar on every page. The Doop frames carry it as the last item in each sidebar (12 screen frames). The wireframes gave no footer spec, so these values were chosen during the build.

```mermaid
flowchart TD
    S[Sidebar] --> B[Brand row: icon and title]
    S --> M[Menu: Chat, History, Documents, Model]
    S --> C[Chat only: model picker and New chat]
    S --> F[Footer: Powered by Vidhaata Labs, pinned to bottom]
```

Decided: the brand row sits above the menu, not below it. Streamlit's native logo cannot hold text, so the title is drawn into the logo image and its font is approximated from Inter Bold, so the rendering may differ slightly from the wireframe type.
