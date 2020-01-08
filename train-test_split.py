from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset dari library sklearn
datasets = load_iris()
data_train = datasets.data
data_labels = datasets.target

# Lakukan split Train dan Test data dari datasets menggunakan fungsi train_test_split
# Test Size dapat disesuaikan sesuai dengan kebutuhan, disini 0.2 berarti 20% dari datasets akan menjadi data test
# Sementara 80% sisanya akan menjadi data train, Lalu untuk random_state digunakan untuk menetapkan hasil
# split data agar selalu sama (berguna seperti menetapkan patokan). Lalu kenapa harus 42 ? pada dasarnya
# bebas untuk menetapkan angka random state, hanya lebih sering orang-orang menggunakan 42.
# Sebenarnya ada alasan dibalik angka 42 tetapi lebih ke alasan atas dasar kepercayaan / myth dan bukan
# ke alasan yang teoritis (bisa cari internet), maka dari itu tidak akan dibahas disini
train_x, test_x, train_y, test_y = train_test_split(data_train, data_labels, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(train_x, train_y)

# Model yang sudah ditrain digunakan untuk memprediksi test data.
# Lalu hasil prediksinya akan dibandingkan dan dicari skore akurasi dari model tersebut.
prediction = model.predict(test_x)
score = accuracy_score(test_y, prediction)
score = score * 100

# Hasil Akurasi Yang didapatkan dalam bentuk persentase
print('Accuracy Score : ' + str(score) + '%')