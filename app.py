from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI untuk mencegah error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'

# Pastikan folder untuk uploads dan images ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['IMAGE_FOLDER']):
    os.makedirs(app.config['IMAGE_FOLDER'])


@app.route('/')
def index():
    return redirect(url_for('upload'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file.filename == '':
            return "No file selected"
        if not file.filename.endswith('.csv'):
            return "Hanya file CSV yang diperbolehkan."
        
        # Simpan file yang diupload
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Membaca dataset
        data = pd.read_csv(filepath)
        return render_template(
            'data_overview.html',
            filename=file.filename,
            data=data.to_html(classes='table table-striped')
        )

    return render_template('upload.html')


@app.route('/elbow/<filename>')
def elbow(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Pilih kolom numerik untuk klasterisasi
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        return "Dataset tidak memiliki kolom numerik untuk analisis."

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # Hitung inertia untuk berbagai nilai k
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Tentukan jumlah klaster optimal (metode elbow)
    knee_locator = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    # Grafik Elbow
    elbow_path = os.path.join(app.config['IMAGE_FOLDER'], 'elbow_plot.png')
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o', linestyle='--')
    plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.savefig(elbow_path)
    plt.close()

    return render_template(
        'elbow.html',
        filename=filename,
        elbow_path=url_for('static', filename='images/elbow_plot.png'),
        optimal_k=optimal_k
    )


@app.route('/clustering/<filename>', methods=['POST'])
def clustering(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Pilih kolom numerik untuk klasterisasi
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        return "Dataset tidak memiliki kolom numerik untuk klasterisasi."
    if len(numeric_data.columns) < 2:
        return "Dataset harus memiliki setidaknya dua kolom numerik untuk visualisasi."

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # Ambil jumlah klaster dari form
    k = int(request.form['k'])

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    # Pisahkan data berdasarkan klaster
    cluster_data = {}
    for cluster in range(k):
        cluster_data[cluster] = data[data['Cluster'] == cluster]

    # Visualisasi hasil clustering
    cluster_path = os.path.join(app.config['IMAGE_FOLDER'], 'cluster_plot.png')
    plt.figure(figsize=(8, 6))
    plt.scatter(data[numeric_data.columns[0]], data[numeric_data.columns[1]], c=data['Cluster'], cmap='viridis', alpha=0.6)
    plt.title(f'Klasterisasi dengan {k} Klaster')
    plt.xlabel(numeric_data.columns[0])
    plt.ylabel(numeric_data.columns[1])
    plt.savefig(cluster_path)
    plt.close()

    return render_template(
        'result.html',
        cluster_path=url_for('static', filename='images/cluster_plot.png'),
        cluster_data=cluster_data
    )


if __name__ == '__main__':
    app.run(debug=True)
