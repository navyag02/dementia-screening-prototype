import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import tempfile
import os
import random
import time
import string
from st_audiorec import st_audiorec

# --- Model & Feature Column Loading ---
# Wrapped in a function with caching for efficiency
@st.cache_resource
def load_model_and_columns():
    try:
        model = joblib.load("mock_hybrid_model.pkl")
        # ASSUMPTION: The model was trained with specific columns after one-hot encoding.
        # We need to know these columns. We'll simulate this by defining them.
        # In a real scenario, you would save these columns during training.
        model_cols = [
            'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
            'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'pauses',
            'task_Free Speech (Audio)', 'task_Memory Recall (Audio)',
            'task_Picture Description (Audio)', 'task_Verbal Fluency (Audio)'
        ]
        return model, model_cols
    except FileNotFoundError:
        st.error("Error: The model file 'mock_hybrid_model.pkl' was not found.")
        st.info("Please make sure the trained model file is in the same directory.")
        return None, None

model, model_columns = load_model_and_columns()

# --- Feature Extractor ---
def extract_features(filepath):
    """Extracts MFCC and pause features from an audio file."""
    y, sr = librosa.load(filepath, sr=16000)

    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Calculate pauses based on energy (removed redundant abs())
    energy = np.array([sum(y[i:i+512]**2) for i in range(0, len(y), 512)])
    pauses = np.sum(energy < np.percentile(energy, 10))

    # Create a dictionary of base features
    features = {f"mfcc_{i}": mfcc_mean[i] for i in range(len(mfcc_mean))}
    features["pauses"] = pauses
    
    return features

# --- Streamlit UI ---
st.title("🧠 Early Dementia Detection Prototype")
st.write("A prototype demonstrating dementia screening with **speech analysis + cognitive tasks**.")

task_type = st.selectbox(
    "Choose a screening task:",
    [
         "Picture Description (Audio)", "Memory Recall (Audio)",
        "Verbal Fluency (Audio)", "Word Recall (Text)", "Math Puzzle (Text)",
        "Category Fluency (Text)"
    ]
)

# --- Audio Tasks ---
if "Audio" in task_type and model is not None:
    if task_type == "Memory Recall (Audio)":
        # --- Test Content ---
        
        # The story and its key points are defined here.
        STORY_TEXT = """
        Anna, a florist from the small town of Greenville, decided to visit the city market on a sunny Tuesday. 
        She took a blue bus, number 42, which arrived at 10 AM. At the market, she bought a basket of red apples and a small silver key for her greenhouse. 
        Later, she met her friend, a musician named Leo, near a large clock tower before returning home in the evening.
        """
        # NOTE: This is a placeholder audio URL. For a real application, you would record the
        # STORY_TEXT above and host the audio file (e.g., on AWS S3, Google Cloud Storage)
        # and replace this URL with the link to your file.
        STORY_AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Human-voice.Ogg"

        KEY_POINTS = [
            "anna", "florist", "greenville", "tuesday", "blue bus", "42", "10 am", 
            "market", "apples", "silver key", "leo", "musician", "clock tower"
        ]

        # --- Session State Initialization ---
        if 'test_stage' not in st.session_state:
            st.session_state.test_stage = 'start'
        if 'distraction_answer' not in st.session_state:
            st.session_state.distraction_answer = 0
        if 'user_audio' not in st.session_state:
            st.session_state.user_audio = None
        if 'score' not in st.session_state:
            st.session_state.score = 0

        # --- Core Functions ---
        def start_new_test():
            """Resets the state for a new test."""
            st.session_state.distraction_answer = random.randint(20, 50) + random.randint(20, 50)
            st.session_state.user_audio = None
            st.session_state.score = 0
            st.session_state.test_stage = 'listen'

        def show_listen_screen():
            """Displays the story for the user to memorize via an audio player."""
            st.subheader("Step 1: Listen and Memorize")
            st.info("Press play on the audio player below and listen carefully to the short story. Try to remember as many details as you can.", icon="🎧")
            
            # Use st.audio to present the story as audio.
            st.audio(STORY_AUDIO_URL)

            with st.expander("View story text (optional)"):
                st.markdown(f"> {STORY_TEXT}")

            st.markdown("---")
            if st.button("I have memorized the story", type="primary"):
                st.session_state.test_stage = 'distraction'
                st.rerun()

        def show_distraction_screen():
            """Shows a simple task to distract the user before recall."""
            st.subheader("Step 2: Quick Brain Teaser")
            st.info("Before you recall the story, please solve this simple math problem.", icon="🧮")
            
            user_answer = st.number_input(f"What is {st.session_state.distraction_answer - 15} + 15?", step=1)

            if st.button("Continue to Recall"):
                if user_answer == st.session_state.distraction_answer:
                    st.session_state.test_stage = 'recall'
                    st.rerun()
                else:
                    st.error("The answer is not quite right, please try again.")

        def show_recall_screen():
            """Prompts the user to record their recollection of the story."""
            st.subheader("Step 3: Recall the Story")
            st.info("Now, please tell me everything you can remember about the story you heard. Include as many details as possible.", icon="🗣️")

            audio_bytes = st_audiorec()
            if audio_bytes:
                st.session_state.user_audio = audio_bytes
                st.success("Your recollection has been recorded.")
            
            st.markdown("---")
            if st.session_state.user_audio:
                st.audio(st.session_state.user_audio)
                if st.button("Analyze My Recollection", type="primary"):
                    st.session_state.test_stage = 'analyzing'
                    st.rerun()

        def show_analysis_screen():
            """Simulates the analysis of the user's audio."""
            st.subheader("Analyzing Your Recollection...")
            st.info(
                "In a real application, an AI model would transcribe your audio and compare it against the key points of the story. "
                "We are simulating this process for the demo.", icon="🤖"
            )

            progress_bar = st.progress(0, text="Processing audio...")
            time.sleep(1)
            progress_bar.progress(50, text="Comparing against key details...")
            time.sleep(1)

            # --- Simulate Scoring ---
            # A real app would use NLU. Here, we generate a realistic random score.
            # We make it more likely to get a higher score to be encouraging.
            recalled_points_count = random.choices(
                population=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                weights=   [1, 1, 2, 3, 4, 5, 4, 3, 2,  1,  1],
                k=1
            )[0]
            st.session_state.score = recalled_points_count

            progress_bar.progress(100, text="Analysis complete!")
            time.sleep(1)
            st.session_state.test_stage = 'finished'
            st.rerun()

        def show_results_screen():
            """Displays the final score and interpretation."""
            st.subheader("Analysis Complete")
            
            score = st.session_state.score
            total_points = len(KEY_POINTS)
            
            st.metric("Memory Recall Score", f"{score} / {total_points} Key Details")

            if score >= 10:
                st.success("Excellent performance! You have a strong ability for narrative recall.", icon="🏆")
            elif 6 <= score < 10:
                st.info("Good performance. You recalled a healthy number of details, which is typical for most adults.", icon="✅")
            else:
                st.warning(
                    "This score is a bit below the average range. Memory can be affected by many things, like focus and sleep. If this is a consistent issue, it might be worth noting.",
                    icon="⚠️"
                )
            
            with st.expander("What a real analysis looks for"):
                st.write("A true AI analysis would check your recording for these key details:")
                st.write(", ".join(f"`{kp}`" for kp in KEY_POINTS))

            st.markdown("---")
            if st.button("Start a New Test", type="primary"):
                start_new_test()
                st.rerun()

       

        if st.session_state.test_stage == 'start':
            if st.button("Begin the Test", type="primary"):
                start_new_test()
                st.rerun()
        elif st.session_state.test_stage == 'listen':
            show_listen_screen()
        elif st.session_state.test_stage == 'distraction':
            show_distraction_screen()
        elif st.session_state.test_stage == 'recall':
            show_recall_screen()
        elif st.session_state.test_stage == 'analyzing':
            show_analysis_screen()
        elif st.session_state.test_stage == 'finished':
            show_results_screen()

    
    
    
    
    if task_type == "Picture Description (Audio)":
        # --- Initialize Session State ---
        if 'app_stage' not in st.session_state:
            st.session_state.app_stage = 'waiting_for_image'
        if 'image_data' not in st.session_state:
            st.session_state.image_data = None
        if 'description_text' not in st.session_state:
            st.session_state.description_text = ""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}

        # --- Core Functions ---

        def show_image_input_screen():
            """Displays the interface for uploading or capturing an image."""
            st.info("Upload an image or take a picture and describe it in a few sentences.", icon="🖼️")

            tab1, tab2 = st.tabs(["📷 Capture / Upload Image", "✏️ Enter Description Manually"])

            with tab1:
                uploaded_file = st.file_uploader(
                    "Upload an image (.jpg, .png)",
                    type=["jpg", "jpeg", "png"]
                )
                if uploaded_file:
                    st.session_state.image_data = uploaded_file.getvalue()
                    st.success("Image uploaded!")
                    st.image(st.session_state.image_data, caption="Uploaded Image", use_container_width=True)

            with tab2:
                description = st.text_area("Or, describe an image from your memory or imagination:")
                if description:
                    st.session_state.description_text = description
                    st.success("Description saved!")

            st.markdown("---")

            if st.session_state.image_data or st.session_state.description_text:
                if st.button("Analyze Description", type="primary"):
                    st.session_state.app_stage = 'analyzing'
                    st.rerun()


        def show_analysis_screen():
            """Simulates analyzing the picture description."""
            st.subheader("Analyzing Description...")
            st.info(
                "In a real application, an AI model could analyze linguistic complexity, content richness, and cognitive patterns. This simulation shows possible metrics.", icon="🤖"
            )

            # Simulate a processing delay
            progress_bar = st.progress(0, text="Processing...")
            for i in range(100):
                time.sleep(0.03)
                progress_bar.progress(i + 1, text=f"Analyzing segment {i+1}...")

            # Generate Simulated Description Metrics
            word_count = len(st.session_state.description_text.split()) if st.session_state.description_text else random.randint(15, 50)
            sentence_count = st.session_state.description_text.count('.') if st.session_state.description_text else random.randint(2, 5)
            keyword_score = random.randint(50, 90)

            st.session_state.analysis_results = {
                "Word Count": word_count,
                "Sentence Count": sentence_count,
                "Keyword Richness (%)": keyword_score
            }

            st.session_state.app_stage = 'finished'
            st.rerun()


        def show_results_screen():
            """Displays the final analysis results and interpretation."""
            st.subheader("Analysis Complete")
            st.write("Here are the simulated metrics from your picture description:")

            results = st.session_state.analysis_results
            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", results['Word Count'])
            col2.metric("Sentence Count", results['Sentence Count'])
            col3.metric("Keyword Richness", f"{results['Keyword Richness (%)']}%")

            with st.expander("How to interpret these results (for demonstration purposes)"):
                st.write("""
                - **Word Count:** More words may indicate richer description, but very high counts may include irrelevant details.
                - **Sentence Count:** More sentences may suggest better organization of thoughts.
                - **Keyword Richness:** A higher score indicates the use of important descriptive or content-related words.
                
                **Disclaimer:** These simulated metrics are for educational purposes only. Real cognitive assessment requires a professional evaluation.
                """)

            st.markdown("---")
            if st.button("Analyze Another Description", type="primary"):
                st.session_state.app_stage = 'waiting_for_image'
                st.session_state.image_data = None
                st.session_state.description_text = ""
                st.session_state.analysis_results = {}
                st.rerun()


        # --- State Machine ---
        if st.session_state.app_stage == 'waiting_for_image':
            show_image_input_screen()
        elif st.session_state.app_stage == 'analyzing':
            show_analysis_screen()
        elif st.session_state.app_stage == 'finished':
            show_results_screen()
    
    if task_type == "Verbal Fluency (Audio)":
        if 'app_stage' not in st.session_state:
            st.session_state.app_stage = 'waiting_for_audio'
        if 'audio_data' not in st.session_state:
            st.session_state.audio_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}

        # --- Core Functions ---

        def show_audio_input_screen():
            """Displays the interface for recording or uploading audio."""
            st.info("Record a short sample of your speech (e.g., describe your day, a recent memory, or what you see in the room) or upload an existing audio file.", icon="💡")

            # --- Audio Input Options ---
            tab1, tab2 = st.tabs(["🎤 Record Live Audio", "⬆️ Upload Audio File"])

            with tab1:
                st.write("Click the microphone icon to start recording. Speak for at least 15-20 seconds for a better analysis. Click the icon again to stop.")
                audio_bytes = st_audiorec()
                if audio_bytes:
                    st.session_state.audio_data = audio_bytes
                    st.success("Recording saved!")

            with tab2:
                uploaded_file = st.file_uploader(
                    "Or, upload an audio file (.wav, .mp3)",
                    type=["wav", "mp3", "m4a"]
                )
                if uploaded_file:
                    st.session_state.audio_data = uploaded_file.getvalue()
                    st.success("File uploaded!")
            
            st.markdown("---")

            # Show the analyze button only if audio data exists
            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data)
                if st.button("Analyze Speech Sample", type="primary"):
                    st.session_state.app_stage = 'analyzing'
                    st.rerun()

        def show_analysis_screen():
            """Simulates the audio analysis process with a progress bar."""
            st.subheader("Analyzing Speech Features...")
            st.info(
                "In a real application, an AI model would analyze acoustic features like speech rate, pause duration, and pitch variation. We are simulating this process.", icon="🤖"
            )
            
            # Simulate a processing delay
            progress_bar = st.progress(0, text="Processing...")
            for i in range(100):
                time.sleep(0.04)
                progress_bar.progress(i + 1, text=f"Analyzing segment {i+1}...")
            
            # --- Generate Simulated Speech Metrics ---
            # These are more descriptive than a single score for free speech.
            st.session_state.analysis_results = {
                "Speech Rate (words/min)": random.randint(110, 165),
                "Pause Count": random.randint(4, 25),
                "Pitch Variation (semitones)": round(random.uniform(1.5, 4.5), 2),
            }
            
            st.session_state.app_stage = 'finished'
            st.rerun()

        def show_results_screen():
            """Displays the final analysis results and interpretation."""
            st.subheader("Analysis Complete")
            st.write("Here are the simulated acoustic features from your speech sample:")

            results = st.session_state.analysis_results
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Speech Rate", f"{results['Speech Rate (words/min)']} WPM")
            col2.metric("Pause Count", results['Pause Count'])
            col3.metric("Pitch Variation", f"{results['Pitch Variation (semitones)']} ST")

            # --- Interpretation of Metrics ---
            with st.expander("How to interpret these results (for demonstration purposes)"):
                st.write("""
                - **Speech Rate:** Refers to the speed of speaking. A significantly slower rate can sometimes be associated with word-finding difficulty or cognitive slowing.
                - **Pause Count:** Frequent or long pauses can indicate hesitation or difficulty retrieving words or formulating thoughts.
                - **Pitch Variation:** Reduced variation in pitch (monotone speech) can sometimes be a marker for conditions like depression or certain neurological changes.
                
                **Disclaimer:** These simulated metrics are for educational purposes only. Real cognitive assessment requires a comprehensive evaluation by a qualified healthcare professional.
                """)

            st.markdown("---")
            if st.button("Analyze Another Audio Sample", type="primary"):
                # Reset the state for a new analysis
                st.session_state.app_stage = 'waiting_for_audio'
                st.session_state.audio_data = None
                st.session_state.analysis_results = {}
                st.rerun()

        

        # --- State Machine: Controls which screen to show ---
        if st.session_state.app_stage == 'waiting_for_audio':
            show_audio_input_screen()
        elif st.session_state.app_stage == 'analyzing':
            show_analysis_screen()
        elif st.session_state.app_stage == 'finished':
            show_results_screen()

# --- Text Tasks ---
elif "Text" in task_type:
   

    if task_type == "Word Recall (Text)":
        if 'test_stage' not in st.session_state:
            st.session_state.test_stage = 'start'
        if 'words_to_recall' not in st.session_state:
            st.session_state.words_to_recall = []

        # --- Word Bank ---
        # A list of common, easily visualizable nouns.
        WORD_BANK = [
            "apple", "river", "chair", "flower", "book", "sun", "table", "music",
            "house", "water", "train", "bread", "bird", "light", "horse", "door",
            "friend", "money", "watch", "color", "forest", "moon", "star", "ship",
            "stone", "glass", "key", "road", "cloud", "dream"
        ]
        WORDS_TO_SHOW = 5

        # --- Functions ---
        def start_new_test():
            """Sets up a new test by selecting random words and resetting the stage."""
            st.session_state.words_to_recall = random.sample(WORD_BANK, WORDS_TO_SHOW)
            st.session_state.test_stage = 'memorize'
            st.session_state.recalled_words_text = "" # Clear previous answers

        def show_memorization_screen():
            """Displays the words for the user to memorize."""
            st.subheader("Step 1: Memorize These Words")
            st.info(f"You will have a moment to memorize the {WORDS_TO_SHOW} words below. Click the button when you are ready to recall them.", icon="🧠")

            # Display words
            words = st.session_state.words_to_recall
            for word in words:
                st.markdown(f"### 🔑 {word.capitalize()}")

            st.markdown("---")
            if st.button("I have memorized the words, proceed to recall", type="primary"):
                st.session_state.test_stage = 'recall'
                st.rerun()

        def show_recall_screen():
            """Displays the input for word recall and calculates the score."""
            st.subheader("Step 2: Recall the Words")
            st.info("Please type all the words you can remember from the list. Separate each word with a space.", icon="✍️")

            recalled_words_text = st.text_area(
                "Enter the words here:",
                height=150,
                key="recalled_words_input"
            )

            # --- Scoring Logic ---
            original_words_set = set(st.session_state.words_to_recall)
            recalled_words_list = [word.strip().lower() for word in recalled_words_text.strip().split() if word]
            recalled_words_set = set(recalled_words_list)

            score = len(original_words_set.intersection(recalled_words_set))
            
            st.markdown("---")
            st.metric(label="Your Score", value=f"{score} / {WORDS_TO_SHOW}")

            # --- Interpretation of the Score ---
            if score >= 4:
                st.success("Excellent performance. This indicates a very strong immediate recall ability.", icon="🏆")
            elif score == 3:
                st.info("Good performance. This is within the typical range for healthy adults.", icon="✅")
            else:
                st.warning(
                    "This score is below the typical range. While many factors can affect memory on a given day (like stress or sleep), consistently low scores might be an area to discuss with a healthcare professional.",
                    icon="⚠️"
                )
            
            # --- Detailed Feedback ---
            with st.expander("See a detailed breakdown of your results"):
                correctly_recalled = list(original_words_set.intersection(recalled_words_set))
                missed_words = list(original_words_set.difference(recalled_words_set))
                
                st.write("**Words you correctly recalled:**")
                if correctly_recalled:
                    st.success(", ".join(sorted(correctly_recalled)))
                else:
                    st.write("None yet.")

                st.write("**Words you missed:**")
                if missed_words:
                    st.error(", ".join(sorted(missed_words)))
                else:
                    st.write("You recalled all the words!")

            st.markdown("---")
            if st.button("Start a New Test"):
                start_new_test()
                st.rerun()


        # --- Main App Logic ---
        # This controls which screen is shown based on the session state.
        if st.session_state.test_stage == 'start':
            if st.button("Begin the Test", type="primary"):
                start_new_test()
                st.rerun()
        elif st.session_state.test_stage == 'memorize':
            show_memorization_screen()
        elif st.session_state.test_stage == 'recall':
            show_recall_screen()
        



    if task_type == "Math Puzzle (Text)":
        st.info("This task tests working memory and calculation speed.")
        # Initialize state to prevent the numbers from changing on every interaction
        if 'math_puzzle' not in st.session_state:
            a, b = random.randint(10, 99), random.randint(10, 50)
            st.session_state.math_puzzle = {"a": a, "b": b, "answer": a + b}
        
        a = st.session_state.math_puzzle['a']
        b = st.session_state.math_puzzle['b']
        correct_answer = st.session_state.math_puzzle['answer']

        st.write(f"Solve this quickly: **{a} + {b} = ?**")
        user_answer = st.number_input("Your Answer:", step=1, value=None, placeholder="Type your answer...")
        
        if user_answer is not None:
            if user_answer == correct_answer:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect.")
        
        if st.button("New Puzzle"):
            del st.session_state.math_puzzle
            st.rerun()

    if task_type == "Category Fluency (Text)":
        TEST_DURATION_SECONDS = 60
        CATEGORIES = ["Animals", "Fruits", "Countries", "Musical Instruments", "Things you find in a kitchen", "Sports"]

        # --- Session State Initialization ---
        # This is crucial to manage the state of the test across reruns.
        if 'test_stage' not in st.session_state:
            st.session_state.test_stage = 'start'
        if 'end_time' not in st.session_state:
            st.session_state.end_time = 0
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        if 'current_category' not in st.session_state:
            st.session_state.current_category = ""

        # --- Functions ---

        def start_new_test():
            """Resets the state and starts a new test."""
            st.session_state.current_category = random.choice(CATEGORIES)
            st.session_state.end_time = time.time() + TEST_DURATION_SECONDS
            st.session_state.user_input = ""
            st.session_state.test_stage = 'running'

        def show_running_test_screen():
            """Displays the main test screen with timer and input box."""
            st.subheader(f"Category: {st.session_state.current_category}")
            st.info(f"You have {TEST_DURATION_SECONDS} seconds to list as many items as you can. Press Enter after each item to add it to the list.", icon="⏱️")
            
            # Create a placeholder for the timer
            timer_placeholder = st.empty()
            
            # Text area for user input
            st.session_state.user_input = st.text_area("Type an item and press Enter...", height=250, key="fluency_input")

            time_left = st.session_state.end_time - time.time()
            
            # --- Timer and Rerun Logic ---
            if time_left > 0:
                # Display the timer with a progress bar
                progress_percentage = int(((TEST_DURATION_SECONDS - time_left) / TEST_DURATION_SECONDS) * 100)
                timer_placeholder.metric("Time Remaining", f"{int(time_left)} seconds")
                st.progress(progress_percentage)
                
                # This part is key: it forces the script to rerun every fraction of a second
                # to update the timer display, creating a live countdown effect.
                time.sleep(0.1)
                st.rerun()
            else:
                # When time is up, show "Time's Up!" and move to the results stage.
                timer_placeholder.metric("Time Remaining", "Time's Up!")
                st.session_state.test_stage = 'finished'
                # A brief pause before showing results.
                time.sleep(1)
                st.rerun()

        def show_results_screen():
            """Calculates and displays the final score and interpretation."""
            st.subheader("Test Complete!")
            st.info(f"You were asked to name items in the category: **{st.session_state.current_category}**")

            # --- Scoring Logic ---
            # Splits the input by any whitespace (including newlines) or commas, and filters out empty strings.
            items = [item.strip() for item in st.session_state.user_input.replace(",", " ").replace("\n", " ").split() if item.strip()]
            score = len(items)
            
            st.metric("Your Score (Total Items Listed)", score)

            # --- Interpretation ---
            if score >= 20:
                st.success("Excellent performance! A high score indicates strong semantic memory and cognitive flexibility.", icon="🏆")
            elif 12 <= score < 20:
                st.info("Good performance. This score is within the typical range for healthy adults.", icon="✅")
            else:
                st.warning(
                    "This score is below the average range. Many factors can influence performance, but if you consistently find this task challenging, it could be worth discussing with a healthcare provider.",
                    icon="⚠️"
                )
            
            # --- Display User's List ---
            with st.expander("See the list of items you entered"):
                if items:
                    st.write(", ".join(items))
                else:
                    st.write("You didn't list any items.")

            st.markdown("---")
            if st.button("Start a New Test", type="primary"):
                start_new_test()
                st.rerun()



        # --- Main App Logic (State Machine) ---
        if st.session_state.test_stage == 'start':
            if st.button("Begin the Test", type="primary"):
                start_new_test()
                st.rerun()
        elif st.session_state.test_stage == 'running':
            show_running_test_screen()
        elif st.session_state.test_stage == 'finished':
            show_results_screen()

