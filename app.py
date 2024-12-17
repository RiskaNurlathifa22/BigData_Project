import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Buat aplikasi Flask
app = Flask(__name__)

# Path file CSV lokal
books_path = os.path.join(os.getcwd(), 'data', 'books.csv')
users_path = os.path.join(os.getcwd(), 'data', 'users.csv')
ratings_path = os.path.join(os.getcwd(), 'data', 'ratings.csv')

# Membaca dataset menggunakan Pandas dengan pemisahan koma
books_df = pd.read_csv(books_path, sep=',', skiprows=1)
users_df = pd.read_csv(users_path, sep=',', skiprows=1)
ratings_df = pd.read_csv(ratings_path, sep=',', skiprows=1)

# Menetapkan nama kolom yang sesuai
books_df.columns = ['Index', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
users_df.columns = ['Index', 'User-ID', 'Location', 'Age']
ratings_df.columns = ['Index', 'User-ID', 'ISBN', 'Book-Rating']

# Memeriksa kolom pada books_df
print("Kolom pada books_df:", books_df.columns)

# Memastikan kolom ISBN ada pada books_df
if 'ISBN' in books_df.columns:
    print("Kolom ISBN ditemukan di books_df.")
else:
    print("Kolom ISBN tidak ditemukan di books_df.")

# Memastikan kolom pada ratings_df sesuai
print("Kolom pada ratings_df:", ratings_df.columns)

# Gabungkan ratings_df dengan books_df berdasarkan kolom 'ISBN'
df = pd.merge(ratings_df, books_df, on='ISBN', how='inner')
df = pd.merge(df, users_df, on="User-ID", how='inner')

# Hapus data yang tidak lengkap
combine_book_rating = df.dropna(axis=0, subset=['Book-Title'])

# Hitung jumlah rating untuk setiap buku
book_ratingCount = combine_book_rating.groupby(by=['Book-Title'])['Book-Rating'].count().reset_index()
book_ratingCount.rename(columns={'Book-Rating': 'totalRatingCount'}, inplace=True)
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='Book-Title', right_on='Book-Title', how='left')

# Filter buku populer
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query(f'totalRatingCount >= {popularity_threshold}')

# Buat Pivot Table
book_features_df = rating_popular_book.pivot_table(values='Book-Rating', index='Book-Title', columns='User-ID', fill_value=0)

# Konversi ke matriks sparse
book_features_df_matrix = csr_matrix(book_features_df.values)

# Inisialisasi model KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_features_df_matrix)

@app.route("/")
def home():
    return render_template("index.html", books=list(book_features_df.index))
print(book_features_df.index)
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Ambil data dari form
        book_title = request.form["book_title"]
        
        # Cek apakah buku ada di dataset
        if book_title not in book_features_df.index:
            return render_template("recommendations.html", error=f"Buku '{book_title}' tidak ditemukan dalam dataset.")

        # Cari tetangga terdekat
        distances, indices = model_knn.kneighbors(
            book_features_df.loc[book_title].values.reshape(1, -1), n_neighbors=6)

        # Hasil rekomendasi
        # recommendations = [
        #     {"title": book_features_df.index[indices.flatten()[i]], "distance": distances.flatten()[i]}
        #     for i in range(1, len(distances.flatten()))
        # ]
        recommendations = sorted(
            [{"title": book_features_df.index[indices.flatten()[i]], "distance": distances.flatten()[i]}
            for i in range(1, len(distances.flatten()))],
            key=lambda x: x['distance'],  # Mengurutkan berdasarkan nilai 'distance'
            reverse=True  # Membalikkan urutan untuk dari besar ke kecil
        )
        
        # Kirim data ke template untuk ditampilkan
        return render_template("recommendations.html", book_title=book_title, recommendations=recommendations)
    except Exception as e:
        return render_template("recommendations.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)