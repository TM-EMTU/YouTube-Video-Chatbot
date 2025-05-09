# YouTube Video Chatbot 🤖

This project is a **YouTube Video Chatbot** that allows users to interact with the transcript of any YouTube video. It uses **Generative AI** (via Gemini) to answer questions based on the video’s transcript, making it a powerful tool for content interaction and retrieval.

---

### 🌟 **Features:**
- **YouTube Transcript Extraction**: Automatically retrieves the transcript of any YouTube video.
- **Chunking & Embedding**: Splits long video transcripts into smaller chunks and embeds them for efficient searching.
- **Contextual Q&A**: Uses **Google Gemini Generative AI** to generate answers based on the provided transcript.
- **Vector Database (FAISS)**: Efficiently stores and retrieves relevant text chunks for Q&A.
- **Clean User Interface**: Built with **Streamlit**, featuring a chat interface where users can ask questions about the video.

---

### 💡 **Tech Stack:**
- **Python**  
- **Streamlit**: For building the user interface.
- **LangChain**: For document processing, chunking, and embeddings.
- **Google Gemini AI**: For contextual Q&A and text generation.
- **YouTube Transcript API**: To retrieve YouTube captions.
- **FAISS**: For efficient vector storage and retrieval.
- **dotenv**: To securely handle API keys.

---

### 🚀 **Installation & Setup:**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/YouTube-Video-Chatbot.git
   cd YouTube-Video-Chatbot
   ```

2. **Install Required Libraries:**

   Install the dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Google API Key:**

   - Create a `.env` file in the root directory of the project and add your Google API Key like so:

     ```
     GEMINI_API_KEY=your-google-api-key
     ```

4. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

---

### 🛠️ **How It Works:**

1. **Input Video ID**: Enter the YouTube video ID into the input box.
2. **Transcript Extraction**: The app retrieves the video transcript using the **YouTube Transcript API**.
3. **Text Chunking**: The transcript is split into manageable chunks for better processing.
4. **Embedding & Search**: The chunks are converted into vector embeddings and stored in **FAISS** for efficient searching.
5. **Ask Questions**: Type your questions in the chat box, and **Gemini AI** generates answers based on the most relevant sections of the transcript.

---

### 🌐 **Demo:**

Check out a live demo of the app here:  
[Demo Link](#)

---

### 👨‍💻 **About the Developer:**

**Tanjil Mahmud Emtu**  
AI Developer | Python Enthusiast | Generative AI Specialist  

**Connect with me:**  
- [GitHub](https://github.com/TM-EMTU/)  
- [LinkedIn]([https://linkedin.com/in/your-link](https://www.linkedin.com/in/tanjil-mahmud-1551aa334/))

---

### 📄 **Licensing:**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 💬 **Contributing:**

1. Fork this repository.
2. Create your feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

### 📝 **Acknowledgements:**

- [LangChain](https://www.langchain.com/) for the powerful tools in document processing and working with LLMs.
- [Google Gemini](https://cloud.google.com/blog/topics/ai-machine-learning) for their cutting-edge AI models.
- [Streamlit](https://streamlit.io/) for simplifying app development.

---

### 📜 **Additional Notes:**

- Ensure your Google API key is valid for accessing the **Gemini** model.
- The app currently supports English transcripts, but can be extended to other languages by modifying the **YouTube Transcript API** language setting.
