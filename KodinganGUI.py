import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from keras.models import load_model
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
import joblib
import time
import librosa
import sounddevice as sd
import soundfile as sf

def extract_features(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        sr = sound.sampling_frequency
        sound_array = sound.values.T.flatten()
        
        augmented_sound = augment_data_single(sound_array, sr)
        augmented_sound = parselmouth.Sound(augmented_sound, sr)
        
        point_process = parselmouth.praat.call(augmented_sound, "To PointProcess (periodic, cc)", 75, 500)
        
        jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0.02, 0.0001, 0.02, 1.3)
        shimmer_local = parselmouth.praat.call([augmented_sound, point_process], "Get shimmer (local)", 0, 0.02, 0.0001, 0.02, 1.3, 1.6)
        hnr = parselmouth.praat.call(augmented_sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean = parselmouth.praat.call(hnr, "Get mean", 0, 0)
        mfcc = augmented_sound.to_mfcc(number_of_coefficients=13).to_array().mean(axis=1)
        
        features = [jitter_local, shimmer_local, hnr_mean] + list(mfcc)
        features = np.nan_to_num(features)  # Replace NaN values with 0
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def augment_data_single(X, sr):
    noise = np.random.normal(0, 0.005, X.shape)
    X_noise = X + noise
    pitch_shift = np.random.randint(-3, 3)
    try:
        X_pitch = librosa.effects.pitch_shift(X.flatten(), sr=sr, n_steps=pitch_shift).reshape(X.shape)
    except librosa.util.exceptions.ParameterError:
        X_pitch = X
    return X_pitch

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(1, -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(1, X.shape[0], 1)

class DepressionDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Sistem Deteksi Depresi")
        master.geometry("650x420")
        master.configure(bg="#4a4a4a")

        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1()

    def show_page1(self):
        self.page2.hide()
        self.page1.show()

    def show_page2(self, prediction_result, confidence):
        self.page1.hide()
        self.page2.show(prediction_result, confidence)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()
        self.frame.configure(bg="#4a4a4a")

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.label.pack(pady=20)

        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 28), bg="white", fg="black", command=self.record_audio)
        self.record_button.pack(pady=20)

        self.choose_file_button = tk.Button(self.frame, text="Pilih File Audio", font=("Helvetica", 28), bg="white", fg="black", command=self.choose_audio_file)
        self.choose_file_button.pack(pady=20)

    def show(self):
        self.frame.pack()

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        self.label.config(text="Perekaman suara...")
        self.master.update()

        duration = 4  # seconds
        fs = 22050  # Sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        audio_path = 'temp_recording.wav'
        sf.write(audio_path, recording.flatten(), fs)
        
        self.label.config(text="Recording finished. Predicting, please wait...")
        self.master.update()
        
        model_path = "D:/Krispi/SimpanModel/best_model_depression.h5"
        scaler_path = 'D:/Krispi/SimpanModel/scaler_depression.pkl'
        predicted_label, prediction = predict_new_audio(audio_path, model_path, scaler_path)
        label_map = {0: "Non-Depressed", 1: "Depressed"}
        result = label_map[predicted_label[0]]
        confidence = prediction[0][predicted_label[0]]

        self.app.show_page2(result, confidence)

    def choose_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                model_path = "D:/Krispi/SimpanModel/best_model_depression.h5"
                scaler_path = 'D:/Krispi/SimpanModel/scaler_depression.pkl'
                print(f"Predicting audio file: {file_path}")
                predicted_label, prediction = predict_new_audio(file_path, model_path, scaler_path)
                label_map = {0: "Non-Depressed", 1: "Depressed"}
                result = label_map[predicted_label[0]]
                confidence = prediction[0][predicted_label[0]]
                self.app.show_page2(result, confidence)
            except Exception as e:
                print(f"Error during prediction: {e}")
                messagebox.showerror("Error", f"An error occurred: {e}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master)
        self.frame.pack()

        self.label = tk.Label(self.frame, text="Hasil Prediksi:", font=("Helvetica", 28), bg="#4a4a4a", fg="white")
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 24), bg="#4a4a4a", fg="white")
        self.confidence_label = tk.Label(self.frame, text="", font=("Helvetica", 20), bg="#4a4a4a", fg="white")

        self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 25), bg="white", fg="black", command=self.app.show_page1)

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=20)
        self.confidence_label.pack(pady=20)
        self.back_button.pack(pady=40)

    def show(self, prediction_result, confidence):
        self.prediction_label.config(text=f"{prediction_result}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        self.frame.pack()
        self.frame.configure(bg="#4a4a4a")

    def hide(self):
        self.frame.pack_forget()

def predict_new_audio(audio_path, model_path, scaler_path):
    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Extracting features from: {audio_path}")
        features = extract_features(audio_path)
        if features is None:
            raise ValueError("Feature extraction failed.")
        print(f"Normalizing features")
        normalized_features = normalize_data(features, scaler)
        print(f"Predicting")
        prediction = model.predict(normalized_features)
        predicted_label = np.argmax(prediction, axis=1)
        return predicted_label, prediction
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise e

if __name__ == "__main__":
    root = tk.Tk()
    app = DepressionDetectionApp(root)
    root.mainloop()
