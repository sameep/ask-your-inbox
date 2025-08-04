# Ask Your Inbox 🤖📧

An application that lets you chat with your entire email archive. Ask questions in natural language and get intelligent, summarized answers from your own MBOX files, with full privacy and local processing.

![Alt Text](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExczhoZHprd2o4cjE3c3JxNmdsajF6cmp5MjZ5a2p6NHlhN3VkcGJrNSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o6gDSdED1B5wjC2Gc/giphy.gif)

---

## Table of Contents

- [What is This?](#what-is-this)
- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Step-by-Step Installation Guide](#step-by-step-installation-guide)
  - [Step 1: Get Your API Keys](#step-1-get-your-api-keys)
  - [Step 2: Get Your MBOX Email Archive](#step-2-get-your-mbox-email-archive)
  - [Step 3: Set Up the Application](#step-3-set-up-the-application)
- [How to Use the App](#how-to-use-the-app)
- [A Note on Privacy](#a-note-on-privacy)
- [License](#license)

---

## What is This?

Your email inbox is a treasure trove of information, but finding anything specific can be a nightmare. Traditional search is often clunky and relies on exact keywords.

**Ask Your Inbox** transforms your email archive into a conversational assistant. It uses powerful AI models to understand the *meaning* and *context* behind your questions, allowing you to find information, summarize conversations, and uncover insights you never thought possible.

This application runs locally on your machine, ensuring your emails remain private.

---

## Features

- **Natural Language Chat**: Ask questions like you would talk to a person, e.g., *"What were the main takeaways from the project alpha meeting last month?"*
- **Advanced Hybrid Search**: Combines semantic (meaning-based) search with keyword, sender, and date filters for highly accurate results.
- **Conversation Thread Analysis**: Understands reply-chains to find relevant context, even if it's buried in a long thread.
- **Local & Private**: Your MBOX email file is processed directly on your computer and is never uploaded to a server.
- **User-Friendly Interface**: A simple web interface powered by Streamlit makes the entire process easy and intuitive.
- **Fast Caching**: After the first processing, your data is saved locally for near-instant re-loading.

---

## How It Works

The app uses a sophisticated multi-stage pipeline to give you the best answers:

1.  **Ingest & Sanitize**: It reads your MBOX file, cleans up the data, and parses it into a structured format.
2.  **Embed**: It uses **Voyage AI** to create numerical representations (embeddings) of your emails based on their meaning.
3.  **Search & Re-rank**: When you ask a question, it filters emails by sender/date, finds the most semantically relevant candidates, and then re-ranks them based on recency, keywords, and conversation context.
4.  **Synthesize**: It sends the most relevant email snippets to **Claude AI** to generate a clear, human-readable summary that directly answers your question.

---

## Prerequisites

Before you begin, you will need the following:

1.  **Python 3.8 or newer**: If you don't have Python, you can download it from the [official Python website](https://www.python.org/downloads/). During installation, make sure to check the box that says **"Add Python to PATH"**.
2.  **API Keys**: You'll need keys from two services to power the AI.
    -   A **Claude AI** API Key for generating answers.
    -   A **Voyage AI** API Key for semantic search.
3.  **An MBOX File**: An archive of your emails.

---

## Step-by-Step Installation Guide

Follow these steps carefully to get the application running.

### Step 1: Get Your API Keys

You need to sign up for two services. Both offer free starting credits that are more than enough to get started.

#### Claude AI (by Anthropic)

1.  Go to the [Anthropic Console](https://console.anthropic.com/) and sign up.
2.  Navigate to the **API Keys** section in your account settings.
3.  Create a new API key and copy it immediately. **You will not be able to see it again!**

#### Voyage AI

1.  Go to the [Voyage AI Dashboard](https://dash.voyageai.com/) and sign up.
2.  Navigate to the **API Keys** section from the left-hand menu.
3.  Your default API key will be visible there. Copy it.

**Keep these two keys safe and ready for a later step.**

### Step 2: Get Your MBOX Email Archive

You need to export your emails into a single `.mbox` file. Here's how to do it for common services:

-   **Gmail**: Use [Google Takeout](https://takeout.google.com/).
    1.  Click "Deselect all".
    2.  Scroll down and select "Mail".
    3.  Choose the option to include all mail or select specific labels.
    4.  Proceed to export. You will receive a link to download a `.zip` file containing your `.mbox` file.
-   **Apple Mail / Thunderbird**: These clients often store emails in MBOX format natively or have a simple export option. Look for an "Export Mailbox" feature.

Save this `.mbox` file somewhere you can easily find it.

### Step 3: Set Up the Application

Now we'll set up the code on your computer. Open your terminal or command prompt to run these commands.

1.  **Clone the Repository**
    This downloads the code to your machine.
    ```bash
    git clone https://github.com/your-username/ask-your-inbox.git
    cd ask-your-inbox
    ```
    *(Replace `your-username/ask-your-inbox` with the actual URL of the repository)*

2.  **Create a Virtual Environment**
    This is a best practice that keeps the app's dependencies isolated.
    ```bash
    python -m venv venv
    ```
    Now, activate it:
    -   On **macOS/Linux**: `source venv/bin/activate`
    -   On **Windows**: `venv\Scripts\activate`

3.  **Install Required Packages**
    This command reads the `requirements.txt` file and installs all the necessary Python libraries.
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use the App

You're all set! Now you can run the application.

1.  **Launch the App**
    In your terminal (make sure your virtual environment is still active), run the following command:
    ```bash
    streamlit run app.py
    ```
    *(Note: The main Python file might have a different name, like `email_chat_app_pro.py`. Use that name if so.)*

2.  **Use the Web Interface**
    Your web browser should automatically open to a new page. If not, your terminal will show you a "Local URL" to visit (e.g., `http://localhost:8501`).

3.  **First-Time Setup in the App:**
    -   In the sidebar on the left, paste your **Voyage AI API Key** and **Claude API Key** into the respective fields and click "Set API Keys".
    -   Click the "Browse files" button to upload your `.mbox` file.
    -   Click the **"Process MBOX File"** button.

4.  **Initial Processing**
    The first time you process a file, the app will analyze and create embeddings for every email. **This may take a while for large archives (several minutes to over an hour).** You will see progress bars for each step.

    The great news is that this is a **one-time process**. The results are cached, so subsequent launches will be lightning-fast.

5.  **Start Chatting!**
    Once processing is complete, the chat interface is ready. Ask anything you want!

    **Example Questions:**
    -   `"Find the flight confirmation email from United last year"`
    -   `"What did Sarah say about the marketing budget in her emails from March?"`
    -   `"Summarize my conversation thread with the new client, Acme Inc."`

---

## A Note on Privacy

Your privacy is paramount.
-   Your `.mbox` file and the generated cache file **never leave your computer**.
-   The only data sent to an external service is the text from the most relevant emails, which is sent to **Claude AI's API** to generate a summarized answer for your specific query. Your broader email archive remains private.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
