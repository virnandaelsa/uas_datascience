from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file.filename == '':
            return "No file selected"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('elbow', filename=file.filename))
    
    return render_template('upload.html')

@app.route('/elbow/<filename>')
def elbow(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    X = data[['Age', 'Spending Score (1-100)']]  # Ganti sesuai kolom dataset Anda
    
    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Metode Elbow
    inertias = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Plot elbow
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o', linestyle='--')
    plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    elbow_path = os.path.join(app.config['IMAGE_FOLDER'], 'elbow_plot.png')
    plt.savefig(elbow_path)
    plt.close()

    return render_template('elbow.html', filename=filename, elbow_path=elbow_path)

@app.route('/clustering/<filename>/<int:k>')
def clustering(filename, k):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)
    X = data[['Age', 'Spending Score (1-100)']]  # Ganti sesuai kolom dataset Anda

    # Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    # Visualisasi hasil clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Age'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis', alpha=0.6)
    plt.title(f'Klasterisasi dengan {k} Klaster')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    cluster_path = os.path.join(app.config['IMAGE_FOLDER'], 'cluster_plot.png')
    plt.savefig(cluster_path)
    plt.close()

    return render_template(
        'clustering.html', 
        cluster_path=cluster_path,
        tables=data.to_html(classes='table table-striped', index=False)
    )

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    app.run(debug=True)
