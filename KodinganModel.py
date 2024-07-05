import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
from keras.utils import to_categorical
from keras.optimizers import Nadam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, SpatialDropout1D, Dense, MaxPooling1D, GlobalAveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay

def extract_features(file_path):
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

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def augment_data_single(X, sr):
    noise = np.random.normal(0, 0.001, X.shape)
    X_noise = X + noise
    pitch_shift = np.random.randint(-1, 1)
    try:
        X_pitch = librosa.effects.pitch_shift(X.flatten(), sr=sr, n_steps=pitch_shift).reshape(X.shape)
    except librosa.util.exceptions.ParameterError:
        X_pitch = X
    return X_pitch

def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
        
        pitch_shift = np.random.randint(-3, 3)
        try:
            x_pitch = librosa.effects.pitch_shift(x.flatten(), sr=22050, n_steps=pitch_shift).reshape(x.shape)
            augmented_X.append(x_pitch)
            augmented_Y.append(y)
        except librosa.util.exceptions.ParameterError:
            pass
        
    return np.array(augmented_X), np.array(augmented_Y)

def build_enhanced_model(input_shape):
    model = Sequential([
        Conv1D(32, padding='same', kernel_size=3, input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.8),
        Conv1D(64, padding='same', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.8),
        Conv1D(128, padding='same', kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.8),
        GlobalAveragePooling1D(),
        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        LeakyReLU(alpha=0.1),
        Dropout(0.8),
        Dense(2, activation='softmax')
    ])
    model.summary()
    optimizer = Nadam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(depressed_folder, non_depressed_folder):
    depressed_paths = [os.path.join(depressed_folder, f) for f in os.listdir(depressed_folder)]
    non_depressed_paths = [os.path.join(non_depressed_folder, f) for f in os.listdir(non_depressed_folder)]
    audio_paths = depressed_paths + non_depressed_paths
    labels = [1] * len(depressed_paths) + [0] * len(non_depressed_paths)
    
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_features, audio_paths))
    
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[1], 1)
    return features, np.array(labels), depressed_paths, non_depressed_paths

def plot_confusion_matrix_and_classification_report(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n", report)
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("F1 Score: ", f1)

if __name__ == '__main__':
    depressed_folder = "D:/Krispi/a/a/depresidata"
    non_depressed_folder = "D:/Krispi/a/a/NonDepression"
    X, Y, depressed_paths, non_depressed_paths = load_data(depressed_folder, non_depressed_folder)
    
    print(f"Total samples loaded: {len(Y)}")
    print(f"Depressed samples: {np.sum(Y == 1)}")
    print(f"Non-depressed samples: {np.sum(Y == 0)}")
    
    le = LabelEncoder()
    dataset_y_encoded = le.fit_transform(Y)
    dataset_y_onehot = to_categorical(dataset_y_encoded)
    X, scaler = normalize_data(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, dataset_y_onehot, test_size=0.2, random_state=42, stratify=dataset_y_onehot)
    
    print(f"Train samples: {len(Y_train)}")
    print(f"Test samples: {len(Y_test)}")
    
    scaler_save_path = 'D:/Krispi/SimpanModel/scaler_depression.pkl'
    joblib.dump(scaler, scaler_save_path)
    
    model = build_enhanced_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("D:/Krispi/SimpanModel/best_model_depression.h5", monitor='val_accuracy', save_best_only=True)
    
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=250, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    
    best_model = tf.keras.models.load_model("D:/Krispi/SimpanModel/best_model_depression.h5")
    evaluation = best_model.evaluate(X_test, Y_test)
    print(f"Test Set Final Loss: {evaluation[0]}")
    print(f"Test Set Final Accuracy: {evaluation[1]}")
    
    Y_pred = best_model.predict(X)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true_classes = np.argmax(dataset_y_onehot, axis=1)
    
    print(f"First 10 predictions: {Y_pred_classes[:10]}")   
    print(f"First 10 true values: {Y_true_classes[:10]}")
    
    class_names = [str(cls) for cls in le.classes_]
    
    plot_confusion_matrix_and_classification_report(Y_true_classes, Y_pred_classes, classes=class_names)
    
    plot_history(history)
