from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import time
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI untuk mencegah error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import chardet

app = Flask(__name__)
app.secret_key = 'your_unique_secret_key_here'  # Set secret key untuk flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'

# Pastikan folder untuk uploads dan images ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['IMAGE_FOLDER']):
    os.makedirs(app.config['IMAGE_FOLDER'])

# Variable global untuk memeriksa status upload
current_file = None  # Variabel untuk menyimpan file yang diupload

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global current_file
    if request.method == 'POST':
        file = request.files['dataset']
        if file.filename == '':
            flash("Tidak ada file yang dipilih. Silakan pilih file untuk diunggah.", 'error')
            return redirect(url_for('upload'))
        if not file.filename.endswith('.csv'):
            flash("Hanya file CSV yang diperbolehkan.", 'error')
            return redirect(url_for('upload'))
        
        # Simpan file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Deteksi encoding
        try:
            import chardet
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read(10000))
                detected_encoding = result['encoding']

            # Baca file dengan encoding terdeteksi
            data = pd.read_csv(filepath, encoding=detected_encoding)
        except UnicodeDecodeError:
            flash("Encoding tidak didukung. Coba unggah file dengan encoding UTF-8 atau Latin1.", 'error')
            return redirect(url_for('upload'))
        except Exception as e:
            flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
            return redirect(url_for('upload'))

        # Simpan nama file
        current_file = file.filename

        # Cek missing values
        if data.isnull().values.any():
            return redirect(url_for('handle_missing', filename=file.filename))

        return render_template(
            'data_overview.html',
            filename=file.filename,
            data=data.to_html(classes='table table-striped')
        )

    return render_template('upload.html')

@app.route('/handle_missing/<filename>', methods=['GET', 'POST'])
def handle_missing(filename):
    global current_file
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))

    if request.method == 'POST':
        # Pilihan metode dari form
        method = request.form.get('method', None)

        if not method:
            flash("Silakan pilih metode untuk menangani missing values.", 'error')
            return redirect(url_for('handle_missing', filename=filename))

        try:
            if method == 'drop_rows':
                data = data.dropna()  # Hapus baris dengan missing values
            elif method == 'drop_columns':
                data = data.dropna(axis=1)  # Hapus kolom dengan missing values
            elif method == 'fill_mean':
                data = data.fillna(data.mean(numeric_only=True))  # Isi dengan mean
            elif method == 'fill_median':
                data = data.fillna(data.median(numeric_only=True))  # Isi dengan median
            elif method == 'fill_mode':
                data = data.fillna(data.mode().iloc[0])  # Isi dengan mode
            else:
                flash("Metode tidak valid.", 'error')
                return redirect(url_for('handle_missing', filename=filename))

            # Simpan dataset yang telah diperbaiki
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            data.to_csv(processed_filepath, index=False)
            flash("Missing values berhasil ditangani.", 'success')
            return redirect(url_for('data_overview'))

        except Exception as e:
            flash(f"Terjadi kesalahan saat menangani missing values: {str(e)}", 'error')
            return redirect(url_for('handle_missing', filename=filename))

    # Jika GET, tampilkan informasi missing values
    missing_info = data.isnull().sum()
    return render_template('handle_missing.html', filename=filename, missing_info=missing_info.to_dict())

@app.route('/data_overview')
def data_overview():
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
    
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))
    
    return render_template('data_overview.html', data=data.to_html(classes='table table-striped'), filename=current_file)

@app.route('/elbow/<filename>', methods=['GET', 'POST'])
def elbow(filename=None):
    global current_file
    filename = filename or current_file  # Gunakan current_file jika filename tidak diberikan
    if filename is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        flash("Dataset tidak memiliki kolom numerik untuk analisis.", 'error')
        return redirect(url_for('upload'))

    # Untuk menghitung grafik Elbow dan saran jumlah klaster optimal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Tentukan jumlah klaster optimal (metode elbow)
    knee_locator = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    elbow_path = os.path.join(app.config['IMAGE_FOLDER'], 'elbow_plot.png')
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o', linestyle='--')
    plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.savefig(elbow_path)
    plt.close()

    unique_id = str(time.time()).replace('.', '')

    return render_template(
        'elbow.html',
        filename=filename,
        elbow_path=url_for('static', filename='images/elbow_plot.png'),
        optimal_k=optimal_k,
        columns=numeric_data.columns.tolist(),  # Pastikan columns dikirimkan dalam bentuk list
        selected_columns=request.form.getlist('selected_columns')  # Mengirimkan kolom yang dipilih
    )

@app.route('/clustering/<filename>', methods=['POST'])
def clustering(filename):
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        flash("Dataset tidak memiliki kolom numerik untuk klasterisasi.", 'error')
        return redirect(url_for('upload'))

    selected_columns = request.form.getlist('selected_columns')
    if not selected_columns:
        flash("Silakan pilih kolom untuk klastering.", 'error')
        return redirect(url_for('elbow', filename=filename))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data[selected_columns])

    k = int(request.form['k'])

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    # Pisahkan data berdasarkan klaster
    cluster_data = {}
    for cluster in range(k):
        cluster_data[cluster] = data[data['Cluster'] == cluster]

    cluster_path = os.path.join(app.config['IMAGE_FOLDER'], 'cluster_plot.png')
    plt.figure(figsize=(8, 6))
    plt.scatter(data[selected_columns[0]], data[selected_columns[1]], c=data['Cluster'], cmap='viridis', alpha=0.6)
    plt.title(f'Klasterisasi dengan {k} Klaster')
    plt.xlabel(selected_columns[0])
    plt.ylabel(selected_columns[1])
    plt.savefig(cluster_path)
    plt.close()

    # Kirimkan cluster_data yang benar sebagai dictionary
    return render_template(
        'result.html',
        cluster_path=url_for('static', filename='images/cluster_plot.png'),
        cluster_data=cluster_data  # Pastikan cluster_data adalah dictionary
    )

@app.route('/visualize/<filename>')
def visualize(filename):
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Select numeric columns and drop the 'Year' column if it exists
    if 'Year' in data.columns:
        numeric_data = data.select_dtypes(include=['float64', 'int64']).drop(columns=['Year'])
    else:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        flash("Dataset tidak memiliki kolom numerik untuk visualisasi.", 'error')
        return redirect(url_for('upload'))

    # Calculate descriptive statistics (mean, median, std, min, max)
    stats = numeric_data.describe().T
    stats['median'] = numeric_data.median()

    # Convert stats to a dictionary for easier access in template
    stats_dict = stats.to_dict(orient='index')

    # Prepare for visualization
    visualize_folder = os.path.join(app.config['IMAGE_FOLDER'], 'visualize')
    if not os.path.exists(visualize_folder):
        os.makedirs(visualize_folder)

    # Hapus file sebelumnya di folder visualize
    for file in os.listdir(visualize_folder):
        os.remove(os.path.join(visualize_folder, file))

    plots = [] 

    # Generate Histogram plots for each column
    for column in numeric_data.columns:
        sanitized_column = column.replace("(", "_").replace(")", "_").replace("\\", "_").replace(" ", "_").replace("/", "_")

        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, color='blue', bins=20)
        plt.title(f'Histogram: {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plot_path = os.path.join(visualize_folder, f'{sanitized_column}_hist.png')
        plt.savefig(plot_path)
        plt.close()
        plots.append(url_for('static', filename=f'images/visualize/{sanitized_column}_hist.png'))

    # Pairwise Scatter Plot (if more than one numeric column)
    if len(numeric_data.columns) >= 2:
        plt.figure(figsize=(8, 6))
        sns.pairplot(numeric_data, diag_kind='kde', plot_kws={'alpha': 0.6})
        scatter_path = os.path.join(visualize_folder, 'scatter_plot.png')
        plt.savefig(scatter_path)
        plt.close()
        plots.append(url_for('static', filename='images/visualize/scatter_plot.png'))

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    heatmap_path = os.path.join(visualize_folder, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    plots.append(url_for('static', filename='images/visualize/heatmap.png'))

    return render_template('visualize.html', plots=plots, columns=numeric_data.columns.tolist(), stats=stats_dict)

if __name__ == '__main__':
    app.run(debug=True) 
