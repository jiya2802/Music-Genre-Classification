import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
import numpy as np
from tensorflow.image import resize

#Function
@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("Trained_model.h5")
  return model

# Loading and preprocessing audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

def model_predict(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements,counts=np.unique(predicted_categories,return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts==max_count]
    return max_elements[0]

#sidebar
# Sidebar heading
st.sidebar.markdown("<h2>🎼 Select Page</h2>", unsafe_allow_html=True)

# Selectbox
app_mode = st.sidebar.selectbox("", ["Home", "About Project", "Prediction"])


## Main Page
if(app_mode=="Home"):


    st.markdown(''' ## Welcome to,\n
    ## GenreSense! 🎶🎧''')
    image_path = "Picture1.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Prediction** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Predition** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
""")
    
#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)

# Prediction Page
elif(app_mode == "Prediction"):
    # Custom styled heading
    st.markdown("""
    <h2 style='text-align: center; color: #4A90E2;font-size: 28px; font-family: "Lucida Console", Monaco, monospace;'>
        ♪ Discover the Magic of Music 🎸
    </h2>
    <p style='text-align: center; font-size: 16px; color: #7D3C98;'>
        Upload an audio file and let the System predict its genre in seconds!
    </p>
    """, unsafe_allow_html=True)
    # st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    if test_mp3 is not None:
        filepath = 'Test_Music/' + test_mp3.name

    # Show Button to play audio
    if st.button("Play Audio"):
        st.audio(test_mp3)

    # Predict Button
    if st.button("Predict"):
        with st.spinner("Please Wait..."):
            X_test = load_and_preprocess_data(filepath)
            with st.spinner('Tuning the beats... 🎵'):
                result_index = model_predict(X_test)

            # Displaying the prediction result
            label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            st.markdown("**:blue[Model Prediction:] It's a :red[{}] music**".format(label[result_index]))

            # Add the custom CSS animation for music notes here:
            st.markdown("""
            <style>
            .note-animation {
                font-size: 50px;
                color: #7D3C98;
                animation: float 5s infinite;
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
                100% { transform: translateY(0px); }
            }
            </style>
            <div class="note-animation">𝄞⨾𓍢ִ໋♬⋆.˚𝄢ᡣ𐭩</div>
            """, unsafe_allow_html=True)
