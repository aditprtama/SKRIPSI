# Sistem Deteksi Depresi Melalui Pengenalan Pola Suara Dengan Mengimplementasikan Metode Voice Quality Analysis

Dataset: https://drive.google.com/drive/folders/1YuIYKUQX5CWDn92mP9KoBkEn9StEGPkX?usp=sharing

Komunikasi merupakan aspek penting dalam kehidupan manusia, dengan ucapan menjadi cara utama untuk mengekspresikan emosi. Emosi memainkan peran krusial dalam interaksi sosial, dan kemampuan untuk mendeteksi emosi seperti depresi melalui suara dapat memberikan kontribusi signifikan dalam memahami kondisi mental seseorang. Depresi adalah gangguan suasana hati yang ditandai dengan perasaan sedih yang mendalam dan kehilangan minat terhadap aktivitas sehari-hari, yang dapat berujung pada penurunan produktivitas, gangguan hubungan sosial, dan keinginan bunuh diri. Menurut Riset Kesehatan Dasar (Riskesdas) 2018, lebih dari 19 juta penduduk Indonesia usia lebih dari 15 tahun memiliki gangguan mental emosional, dengan lebih dari 12 juta di antaranya mengalami depresi. Oleh karena itu, pengembangan teknologi pendeteksi emosi depresi berbasis suara menjadi sangat penting. Penelitian ini bertujuan untuk mengembangkan sistem pendeteksi emosi depresi berbasis suara menggunakan metode Voice Quality Analysis (VQA) yang dengan parameter jitter, shimmer, dan Harmonics-to-Noise Ratio (HNR). Metodologi penelitian ini meliputi analisis, perancangan, implementasi, dan pengujian sistem deteksi depresi menggunakan Raspberry Pi 4B. Proses pengujian melibatkan evaluasi kinerja model Convolutional Neural Network (CNN) yang dibangun berdasarkan fitur-fitur suara hasil ekstraksi Voice Quality Analysis (VQA).




Hasil pengujian menunjukkan bahwa sistem mampu mengklasifikasikan suara depresi dengan akurasi 97% dan F1-score 97%. Gambar dibawah ini menggambarkan confusion matrix dari pengujian model yang menampilkan nilai true label dan predicted label. Angka dalam kotak terang mengindikasikan bahwa model berfungsi dengan baik. Pada confusion matrix, sumbu x merepresentasikan nilai prediksi (0 dan 1), sedangkan sumbu y mewakili nilai kelas asli. Nilai 0 menunjukkan non-depresi dan 1 menunjukkan depresi. Setelah memperoleh nilai true label dan predicted label, dilakukan perhitungan untuk mendapatkan nilai akurasi, recall, dan presisi. Perhitungan akurasi digunakan untuk menentukan seberapa baik model dalam memprediksi setiap sampel data. Presisi dan recall digunakan untuk mengukur ketepatan model dalam menangani kelas positif. Presisi mengukur sejauh mana model dapat mengenali kelas positif tanpa memberikan hasil yang salah, sedangkan recall mengukur sejauh mana model dapat menemukan semua instan kelas positif (menghindari false negative).Kombinasi kedua metrik ini diwakili oleh F1-score dan dengan menggunakan nilai F1-score, kita dapat menilai kemampuan keseluruhan model dalam menjaga tingkat ketelitian dan menemukan semua instans positif. Nilai komposit ini memberikan gambaran menyeluruh tentang kinerja model.

![image](https://github.com/user-attachments/assets/710a96eb-201a-4747-9634-a7651edf8291)

Hasil dari perhitungan akurasi, recall, presisi, dan F1-score terdapat pada tabel dibawah ini

![image](https://github.com/user-attachments/assets/43bcce65-3912-4847-aa41-42157be13b11)


Implementasi pada perangkat keras raspberry pi juga menunjukkan hasil yang baik, dengan tingkat keberhasilan prediksi suara non-depresi sebesar 85% dan suara depresi sebesar 75%.

![image](https://github.com/user-attachments/assets/77c51522-238c-42a9-802d-0885754f67b0)

Penelitian ini diharapkan dapat memberikan kontribusi dalam pengembangan teknologi deteksi emosi, khususnya dalam mendeteksi emosi depresi melalui Voice Quality Analysis (VQA). Selain itu, hasil penelitian ini dapat digunakan sebagai referensi untuk penelitian selanjutnya dalam bidang yang sama.
