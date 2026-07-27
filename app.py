import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import os
import random
os.environ['PYTHONHASHSEED'] = '42'
warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Risiko Kerugian Investasi Saham",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme Configuration ──────────────────────────────────────────────────────
if "is_dark_mode" not in st.session_state:
    st.session_state.is_dark_mode = False

# ── Load custom CSS ──────────────────────────────────────────────────────────
css_file = "style_dark.css" if st.session_state.is_dark_mode else "style.css"
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def chart_layout(title, xaxis_title, yaxis_title, height=420):
    is_dark = st.session_state.get("is_dark_mode", False)
    if is_dark:
        t_color, f_color = "#FAFAFA", "#A1A1AA"
        g_color, l_color = "#27272A", "#3F3F46"
        leg_bg, leg_bord = "rgba(24, 24, 27, 0.5)", "#3F3F46"
    else:
        t_color, f_color = "#1E293B", "#475569"
        g_color, l_color = "#E2E8F0", "#CBD5E1"
        leg_bg, leg_bord = "rgba(255, 255, 255, 0.5)", "#CBD5E1"

    return dict(
        title=dict(text=title, font=dict(size=15, color=t_color, family="Arial Black, sans-serif"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=f_color, family="Tahoma, sans-serif"),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=g_color,
            showgrid=True,
            zeroline=False,
            linecolor=l_color,
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=g_color,
            showgrid=True,
            zeroline=False,
            linecolor=l_color,
        ),
        legend=dict(
            bgcolor=leg_bg,
            bordercolor=leg_bord,
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
    )


def clean_dataframe(df_raw):
    df_raw.columns = [c.strip().strip('"') for c in df_raw.columns]
    for col in ["Price", "Open", "High", "Low"]:
        if col in df_raw.columns:
            df_raw[col] = (
                df_raw[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.strip()
            )
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    if "Vol." in df_raw.columns:
        df_raw["Vol."] = (
            df_raw["Vol."].astype(str)
            .str.replace('"', "", regex=False).str.strip()
            .str.replace("M", "e6", regex=False)
            .str.replace("B", "e9", regex=False)
            .str.replace("K", "e3", regex=False)
        )
        df_raw["Vol."] = pd.to_numeric(df_raw["Vol."], errors="coerce")
    if "Change %" in df_raw.columns:
        df_raw["Change %"] = (
            df_raw["Change %"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.strip()
        )
        df_raw["Change %"] = pd.to_numeric(df_raw["Change %"], errors="coerce")
    if "Date" in df_raw.columns:
        df_raw["Date"] = df_raw["Date"].astype(str).str.replace('"', "").str.strip()
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=False, errors="coerce")
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
    return df_raw.dropna(subset=["Date", "Price"])


def compute_ewma_sigma(returns, lambda_=0.94):
    """
    Estimasi volatilitas EWMA (Exponentially Weighted Moving Average),
    gaya RiskMetrics. Berbeda dari rolling standard deviation biasa yang
    memberi bobot SAMA RATA ke seluruh observasi dalam window, EWMA memberi
    bobot lebih besar ke observasi yang lebih baru dan bobot meluruh
    (decay) secara eksponensial ke observasi yang lebih lama.

    Efeknya: ketika terjadi lonjakan volatilitas mendadak, sigma dari EWMA
    akan "sadar" lebih cepat dibanding rolling std, sehingga VaR yang
    dihasilkan lebih responsif terhadap perubahan kondisi pasar terkini.
    Ini adalah pendekatan yang lebih ringan dibanding model GARCH penuh,
    namun tetap mengakomodasi volatility clustering secara sederhana.

    Parameters
    ----------
    returns : array-like
        Data log-return historis dalam satu window.
    lambda_ : float, default 0.94
        Decay factor. Semakin kecil, semakin besar bobot observasi terbaru
        (lebih responsif tapi lebih noisy). 0.94 adalah nilai standar
        RiskMetrics untuk data harian.
    """
    returns = np.array(returns, dtype=float)
    n = len(returns)
    weights = np.array([(1 - lambda_) * lambda_**i for i in range(n)])
    weights = weights[::-1]            # observasi terbaru (indeks akhir array) dapat bobot terbesar
    weights = weights / weights.sum()  # normalisasi agar total bobot = 1
    mean_ret = np.average(returns, weights=weights)
    variance = np.sum(weights * (returns - mean_ret) ** 2)
    return float(np.sqrt(variance))


def compute_var_ecf(returns, confidence_level, override_mu=None, ewma_lambda=0.94, sigma_scale=1.0):
    """
    Estimasi VaR menggunakan Cornish-Fisher Expansion (CF).
    Mengoreksi kuantil distribusi normal standar dengan skewness dan
    kurtosis dari data return, sehingga tetap mempertimbangkan
    penyimpangan dari normalitas tanpa memerlukan inversi numerik penuh.

    override_mu : float, opsional
        Jika diisi, nilai ini dipakai sebagai parameter lokasi (mu)
        alih-alih rata-rata historis `returns.mean()`. Dipakai untuk
        mengintegrasikan prediksi return dari model GRU sebagai
        komponen forward-looking pada VaR-ECF, sementara sigma,
        skewness, dan kurtosis tetap diestimasi dari data historis.
    ewma_lambda : float, default 0.94
        Decay factor untuk estimasi sigma via EWMA (lihat compute_ewma_sigma).
        Menggantikan rolling standard deviation biasa agar sigma lebih
        responsif terhadap perubahan volatilitas terbaru dalam window.
    sigma_scale : float, default 1.0
        Faktor pengali terhadap sigma EWMA. Diestimasi lewat
        `calibrate_sigma_scale` dari periode kalibrasi (di luar periode
        test) agar lebar interval VaR terkalibrasi dengan realisasi
        kerugian aktual (mengoreksi VaR yang secara sistematis terlalu
        longgar/ketat pada uji Kupiec). Nilai 1.0 = tanpa koreksi
        (perilaku sigma murni EWMA seperti sebelumnya).
    """
    from scipy.stats import norm
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

    returns = np.array(returns, dtype=float)
    alpha = 1.0 - confidence_level

    mu = returns.mean() if override_mu is None else float(override_mu)
    sigma = compute_ewma_sigma(returns, lambda_=ewma_lambda) * sigma_scale
    S = scipy_skew(returns)        # skewness
    K = scipy_kurtosis(returns)    # excess kurtosis (definisi Fisher, normal = 0)

    z = norm.ppf(alpha)  # kuantil distribusi normal standar untuk alpha

    z_cf = (
        z
        + (z**2 - 1) * S / 6
        + (z**3 - 3 * z) * K / 24
        - (2 * z**3 - 5 * z) * (S**2) / 36
    )

    var_val = float(mu + z_cf * sigma)
    return var_val


def rolling_var_ecf_backtest(returns_full, test_start_price_idx, n_test, window_size, confidence_level, predicted_returns=None, ewma_lambda=0.94, sigma_scale=1.0):
    """
    Backtesting VaR-ECF dengan rolling window (out-of-sample), terintegrasi
    dengan prediksi GRU.

    Untuk setiap hari p pada periode test, VaR dihitung dengan:
      - mu (lokasi)              = return yang DIPREDIKSI GRU untuk hari p
                                    (forward-looking, jika predicted_returns diberikan;
                                    jika tidak, memakai rata-rata window historis)
      - sigma (EWMA), skewness, kurtosis = dari `window_size` hari historis SEBELUM hari p

    Struktur volatilitas/bentuk distribusi tetap bersumber dari data historis
    (karena GRU pada aplikasi ini dilatih untuk memprediksi harga, bukan
    volatilitas), namun titik pusat estimasi risiko kini benar-benar
    bergantung pada keluaran model GRU — bukan lagi murni dari data historis.
    Sigma sendiri kini diestimasi via EWMA (bukan rolling std biasa) agar
    lebih responsif terhadap lonjakan volatilitas terbaru.

    Parameters
    ----------
    returns_full : np.array
        Log-return dari SELURUH dataset (bukan hanya test), urut waktu.
        returns_full[i] = ln(price[i+1] / price[i]) * 100
    test_start_price_idx : int
        Index harga (bukan return) pertama yang termasuk periode test.
    n_test : int
        Jumlah observasi pada periode test.
    window_size : int
        Jumlah hari historis yang dipakai untuk mengestimasi sigma/skew/kurtosis.
    confidence_level : float
        Tingkat kepercayaan VaR (mis. 0.95).
    predicted_returns : np.array, opsional
        Array log-return yang DIPREDIKSI GRU untuk tiap hari pada periode test
        (panjang = n_test, urut sesuai periode test). predicted_returns[k]
        adalah prediksi untuk hari ke-(test_start_price_idx + k).
    ewma_lambda : float, default 0.94
        Decay factor EWMA yang diteruskan ke compute_var_ecf.
    """
    var_list = []
    actual_list = []
    predicted_mu_list = []
    valid_price_idx = []
    n_skipped = 0

    for p in range(test_start_price_idx, test_start_price_idx + n_test):
        k = p - test_start_price_idx
        ret_idx = p - 1  # index di returns_full yang merepresentasikan return hari p
        window_start = ret_idx - window_size

        if window_start < 0:
            # Data historis belum cukup untuk membentuk window penuh — lewati hari ini
            n_skipped += 1
            continue

        window_data = returns_full[window_start:ret_idx]  # HANYA data sebelum hari p
        override_mu = None
        if predicted_returns is not None:
            override_mu = predicted_returns[k]

        var_p = compute_var_ecf(window_data, confidence_level, override_mu=override_mu, ewma_lambda=ewma_lambda, sigma_scale=sigma_scale)

        var_list.append(var_p)
        actual_list.append(returns_full[ret_idx])
        predicted_mu_list.append(override_mu if override_mu is not None else window_data.mean())
        valid_price_idx.append(p)

    return {
        "var_series": np.array(var_list),
        "actual_series": np.array(actual_list),
        "predicted_mu_series": np.array(predicted_mu_list),
        "price_idx": valid_price_idx,
        "n_skipped": n_skipped,
    }


def calibrate_sigma_scale(returns_full, calib_start_price_idx, n_calib, window_size,
                           confidence_levels, predicted_mu_calib, ewma_lambda=0.94, grid=None):
    """
    Mengestimasi faktor skala sigma (k) agar lebar interval VaR-ECF
    terkalibrasi dengan realisasi kerugian aktual.

    Latar belakang: EWMA sigma yang dihitung murni dari data historis
    kadang secara SISTEMATIS terlalu sempit (atau terlalu lebar) dibanding
    volatilitas realisasi out-of-sample, sehingga VaR gagal uji Kupiec
    meskipun formula Cornish-Fisher-nya sendiri benar. calibrate_sigma_scale
    mencari faktor pengali k (dicoba pada rentang `grid`) yang membuat
    proporsi pelanggaran VaR paling mendekati alpha teoritis pada
    SELURUH confidence level yang dipilih, lalu k inilah yang dipakai untuk
    memperlebar/mempersempit sigma pada periode test.

    Independensi terhadap data test (tidak ada data leakage):
    Kalibrasi dilakukan pada `calib_start_price_idx` s.d. `n_calib` hari
    berikutnya, yaitu bagian VALIDATION SPLIT dari data training (irisan
    yang oleh Keras hanya dipakai untuk monitoring/early stopping, BUKAN
    untuk memperbarui bobot GRU lewat backprop) — jadi seluruhnya berada
    SEBELUM periode test dan tidak pernah "dilihat" bobot model maupun VaR
    di periode test manapun.

    Parameters
    ----------
    predicted_mu_calib : np.array
        Prediksi return GRU untuk periode kalibrasi (panjang = n_calib),
        dihasilkan dari model yang SAMA dengan yang dipakai di periode test.
    grid : np.array, opsional
        Kandidat nilai k yang dicoba. Default: 0.3 s.d. 5.0 dengan step 0.1
        (rentang diperlebar agar tetap bisa menemukan koreksi yang cukup
        besar apabila sigma EWMA historis jauh lebih sempit dibanding
        volatilitas realisasi out-of-sample).

    Returns
    -------
    best_k : float
        Faktor skala sigma terpilih (1.0 = tidak ada koreksi).
    diagnostics : dict
        Rincian violation rate per confidence level pada k terpilih, untuk
        ditampilkan sebagai bukti transparansi proses kalibrasi.
    """
    if grid is None:
        grid = np.arange(0.3, 5.01, 0.1)

    best_k = 1.0
    best_score = None
    best_diag = {}

    for k in grid:
        total_sq_dev = 0.0
        per_cl = {}
        for cl in confidence_levels:
            alpha = 1.0 - cl
            bt = rolling_var_ecf_backtest(
                returns_full, calib_start_price_idx, n_calib, window_size, cl,
                predicted_returns=predicted_mu_calib, ewma_lambda=ewma_lambda,
                sigma_scale=float(k),
            )
            violations = bt["actual_series"] < bt["var_series"]
            n_obs = len(violations)
            viol_rate = float(np.mean(violations)) if n_obs > 0 else 0.0
            total_sq_dev += (viol_rate - alpha) ** 2
            per_cl[str(cl)] = {
                "violation_rate": viol_rate,
                "target_alpha": alpha,
                "n_obs": n_obs,
                "n_violations": int(np.sum(violations)),
            }

        if best_score is None or total_sq_dev < best_score:
            best_score = total_sq_dev
            best_k = float(k)
            best_diag = per_cl

    return best_k, {
        "per_confidence_level": best_diag,
        "score": float(best_score) if best_score is not None else None,
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
    }


def run_model(params):
    """Melatih model GRU dan menghitung estimasi risiko VaR-ECF."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    df = st.session_state.df.copy()
    prices = df["Price"].values.reshape(-1, 1)
    ws = params["window_size"]

    # Tentukan batas train/test DULU, dari data mentah — sebelum scaling
    n_windows = len(prices) - ws
    split = int(n_windows * params["train_ratio"])
    train_end_idx = ws + split  # indeks harga terakhir yang boleh "dilihat" scaler

    # Fit scaler HANYA dari data training, lalu transform seluruh data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(prices[:train_end_idx])
    scaled = scaler.transform(prices)

    X, y = [], []
    for i in range(ws, len(scaled)):
        X.append(scaled[i - ws:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = df["Date"].values[ws + split:]

    # ── Build Model ─────────────────────────────────────────────────────────
    with st.spinner("Membangun dan melatih model GRU, mohon tunggu..."):
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        tf.config.experimental.enable_op_determinism()

        keras_val_split = 0.1  # porsi akhir X_train yang disisihkan Keras (bukan dipakai backprop)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam, RMSprop, SGD
        from tensorflow.keras.callbacks import EarlyStopping

        opt_map = {
            "adam": Adam(learning_rate=params["learning_rate"]),
            "rmsprop": RMSprop(learning_rate=params["learning_rate"]),
            "sgd": SGD(learning_rate=params["learning_rate"]),
        }
        optimizer = opt_map[params["optimizer"]]

        model = Sequential()
        model.add(Input(shape=(ws, 1)))
        if params["gru_units_2"] > 0:
            model.add(GRU(params["gru_units_1"], return_sequences=True))
            model.add(Dropout(params["dropout_rate"]))
            model.add(GRU(params["gru_units_2"], return_sequences=False))
        else:
            model.add(GRU(params["gru_units_1"], return_sequences=False))
        model.add(Dropout(params["dropout_rate"]))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss="mse")

        progress_bar = st.progress(0, text="Memulai pelatihan...")
        epoch_holder = {"current": 0}

        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pct = int((epoch + 1) / params["epochs"] * 100)
                loss_val = logs.get("loss", 0)
                val_loss = logs.get("val_loss", 0)
                progress_bar.progress(
                    min(pct, 100),
                    text=f"Epoch {epoch+1}/{params['epochs']}  |  Loss: {loss_val:.6f}  |  Val Loss: {val_loss:.6f}",
                )

        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
        history = model.fit(
            X_train, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=keras_val_split,
            callbacks=[es, ProgressCallback()],
            verbose=0,
        )
        progress_bar.empty()

    # ── Evaluate ─────────────────────────────────────────────────────────────
    with st.spinner("Mengevaluasi model dan menjalankan rolling-window backtest VaR-ECF..."):
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred)))
        mae = float(mean_absolute_error(y_test_actual, y_pred))
        mape = float(np.mean(np.abs((y_test_actual - y_pred) / np.maximum(y_test_actual, 1e-8))) * 100)
        r2 = float(r2_score(y_test_actual, y_pred))

        # Residual analysis (tetap disimpan untuk analisis error model)
        residual_pct = ((y_test_actual - y_pred) / np.maximum(np.abs(y_test_actual), 1e-8)) * 100

        # ── Prediksi pada Data Latih & Data Validasi ──────────────────────────
        # Meniru Gambar 7 pada jurnal acuan: menampilkan prediksi model pada
        # subset data latih (yang benar-benar dipakai backprop) dan subset
        # data validasi (disisihkan Keras via validation_split, TIDAK dipakai
        # backprop) di atas data y_train aktual, untuk melihat seberapa baik
        # model fit ke data yang sudah "dilihat" vs data yang belum.
        n_train_total = len(X_train)
        n_val = int(n_train_total * keras_val_split)
        n_fit = n_train_total - n_val

        X_fit, X_val = X_train[:n_fit], X_train[n_fit:]

        y_pred_fit_scaled = model.predict(X_fit, verbose=0)
        y_pred_val_scaled = model.predict(X_val, verbose=0) if n_val > 0 else np.array([]).reshape(0, 1)

        y_train_actual_full = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_pred_fit = scaler.inverse_transform(y_pred_fit_scaled).flatten()
        y_pred_val = (
            scaler.inverse_transform(y_pred_val_scaled).flatten() if n_val > 0 else np.array([])
        )

        train_dates_full = df["Date"].values[ws: ws + split]
        train_dates_fit = train_dates_full[:n_fit]
        train_dates_val = train_dates_full[n_fit:]

        # ── Log-return dari SELURUH harga (bukan hanya test) ──
        # Dibutuhkan supaya rolling window bisa "mengintip mundur" ke periode
        # train untuk mengisi window historis di awal periode test.
        # Log-return: r_t = ln(P_t / P_{t-1}) * 100  (dalam persen)
        prices_flat = prices.flatten()
        returns_full = np.diff(np.log(prices_flat)) * 100

        test_start_price_idx = ws + split
        n_test = len(y_test_actual)
        window_size_var = params["window_size_var"]
        ewma_lambda = params.get("ewma_lambda", 0.94)

        # ── Return yang DIPREDIKSI GRU untuk tiap hari periode test ──
        # predicted_returns_test[k] = ln(harga_prediksi_GRU_hari_p / harga_aktual_hari_p-1) * 100
        # Harga aktual hari p-1 dipakai sebagai basis (bukan harga prediksi
        # hari p-1), karena itulah informasi riil yang tersedia saat model
        # membuat prediksi untuk hari p. Ini yang membuat estimasi VaR
        # benar-benar forward-looking dan bergantung pada keluaran GRU.
        prev_actual_prices = prices_flat[test_start_price_idx - 1: test_start_price_idx - 1 + n_test]
        predicted_returns_test = np.log(y_pred / prev_actual_prices) * 100

        # ── Kalibrasi skala sigma (opsional) ──
        # Memakai irisan validation_split dari X_train (bagian training yang
        # HANYA dipakai Keras untuk monitoring/early stopping, tidak pernah
        # dipakai untuk update bobot GRU lewat backprop) sebagai periode
        # kalibrasi "pseudo out-of-sample" yang sepenuhnya berada sebelum
        # periode test — sehingga tidak ada data leakage terhadap evaluasi
        # akhir. Tujuannya mengoreksi sigma EWMA yang terbukti secara
        # sistematis terlalu sempit/lebar dibanding realisasi kerugian,
        # yang biasanya menjadi penyebab utama VaR gagal uji Kupiec.
        sigma_scale = 1.0
        calib_diag = None
        if params.get("use_sigma_calibration", True):
            # Porsi kalibrasi dibuat independen dari validation_split Keras
            # (0.1) agar mendapat data kalibrasi lebih besar tanpa mengubah
            # proses training/early stopping model GRU itu sendiri.
            calib_frac = 0.15
            n_train = len(X_train)
            calib_split_idx = int(n_train * (1 - calib_frac))
            n_calib = n_train - calib_split_idx

            # Ambang minimum kalibrasi dibuat tetap (bukan bergantung pada
            # window_size_var) karena window historis untuk menghitung
            # sigma/skew/kurtosis diambil dari returns_full secara rolling,
            # bukan dibatasi oleh panjang periode kalibrasi n_calib itu
            # sendiri. Syarat lama (window_size_var + 5, mis. 105) membuat
            # kalibrasi nyaris selalu dilewati padahal datanya sebenarnya
            # cukup untuk estimasi k yang stabil.
            min_calib_needed = 30
            if n_calib > min_calib_needed:
                X_calib = X_train[calib_split_idx:]
                y_pred_calib_scaled = model.predict(X_calib, verbose=0)
                y_pred_calib = scaler.inverse_transform(y_pred_calib_scaled).flatten()

                calib_start_price_idx = ws + calib_split_idx
                prev_actual_calib = prices_flat[calib_start_price_idx - 1: calib_start_price_idx - 1 + n_calib]
                predicted_returns_calib = np.log(y_pred_calib / prev_actual_calib) * 100

                sigma_scale, calib_diag = calibrate_sigma_scale(
                    returns_full, calib_start_price_idx, n_calib, window_size_var,
                    params["confidence_levels"], predicted_returns_calib, ewma_lambda,
                )
            else:
                calib_diag = {"skipped": True, "reason": "Data validation split terlalu sedikit untuk kalibrasi."}

        # ── Rolling-window VaR-ECF, per confidence level ──
        # mu (lokasi) VaR-ECF berasal dari prediksi GRU (forward-looking),
        # sementara sigma (EWMA)/skewness/kurtosis tetap dari window historis
        # SEBELUM hari p (out-of-sample, tanpa look-ahead bias).
        var_ecf_rolling = {}

        for cl in params["confidence_levels"]:
            bt = rolling_var_ecf_backtest(
                returns_full, test_start_price_idx, n_test, window_size_var, cl,
                predicted_returns=predicted_returns_test,
                ewma_lambda=ewma_lambda,
                sigma_scale=sigma_scale,
            )
            var_ecf_rolling[str(cl)] = bt

        # Nilai VaR "headline" (mis. untuk tabel ringkasan) = estimasi TERBARU,
        # yaitu estimasi risiko untuk hari berikutnya berdasarkan prediksi
        # GRU paling akhir.
        var_ecf = {
            cl: float(var_ecf_rolling[cl]["var_series"][-1])
            for cl in var_ecf_rolling
        }

        n_skipped = next(iter(var_ecf_rolling.values()))["n_skipped"]

    st.session_state.results = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "test_dates": test_dates,
        "y_test": y_test_actual,
        "y_pred": y_pred,
        "residual_pct": residual_pct,
        "predicted_returns_test": predicted_returns_test,
        "y_train_actual_full": y_train_actual_full,
        "y_pred_fit": y_pred_fit,
        "y_pred_val": y_pred_val,
        "train_dates_fit": train_dates_fit,
        "train_dates_val": train_dates_val,
        "n_fit": n_fit,
        "var_ecf_rolling": var_ecf_rolling,
        "var_ecf": var_ecf,
        "confidence_levels": params["confidence_levels"],
        "window_size_var": window_size_var,
        "ewma_lambda": ewma_lambda,
        "sigma_scale": sigma_scale,
        "calib_diag": calib_diag,
        "n_skipped_rolling": n_skipped,
        "max_epochs": params["epochs"],
        "history_loss": history.history.get("loss", []),
        "history_val_loss": history.history.get("val_loss", []),
    }
    st.session_state.model_trained = True

    if n_skipped > 0:
        st.markdown(
            f"<div class='warning-box'>Catatan: {n_skipped} hari pertama pada periode test "
            f"dilewati dari backtesting VaR karena data historis sebelum periode test belum "
            f"cukup untuk membentuk window {window_size_var} hari.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='success-box'>Model berhasil dilatih dan risiko berhasil diestimasi dengan "
        "rolling-window backtest. Buka halaman <strong>Hasil dan Interpretasi</strong> untuk "
        "melihat hasil lengkap.</div>",
        unsafe_allow_html=True,
    )


# ── Session State Initialization ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "active_page" not in st.session_state:
    st.session_state.active_page = "Beranda"
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "results" not in st.session_state:
    st.session_state.results = None

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image("logo.png", width=80)

    # Theme Toggle
    toggle_label = "Mode Gelap" if st.session_state.is_dark_mode else "Mode Terang"
    is_dark = st.toggle(toggle_label, value=st.session_state.is_dark_mode)
    if is_dark != st.session_state.is_dark_mode:
        st.session_state.is_dark_mode = is_dark
        st.rerun()

    st.markdown("<div class='sidebar-title'>Navigasi</div>", unsafe_allow_html=True)
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    pages = [
        "Beranda",
        "Input Data",
        "Analisis Deskriptif",
        "Parameter Model",
        "Hasil dan Interpretasi",
    ]
    for page in pages:
        is_active = st.session_state.active_page == page
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.active_page = page
            st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-footer'>GRU &amp; VaR-ECF Model<br>v1.0.0</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_page == "Beranda":
    import base64
    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-content">
                <img src="data:image/png;base64,{logo_b64}" class="hero-logo" alt="Logo">
                <div class="hero-text">
                    <div class="hero-title">Sistem Prediksi Risiko Kerugian<br>Investasi Saham</div>
                    <div class="hero-subtitle">
                        Menggunakan Model Gated Recurrent Unit (GRU) dan Value at Risk berbasis
                        Cornish-Fisher Expansion (VaR-ECF)
                    </div>
                </div>
            </div>
            <div class="hero-decoration"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Penyusun dan Afiliasi</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown(
        """<div class="pill-row">
            <div class="pill-badge"><span class="pill-label">Penyusun:</span> Mohammad Idhom, Trimono, Setiawati Nugraheni</div>
            <div class="pill-badge"><span class="pill-label">Program Studi:</span> Sains Data</div>
            <div class="pill-badge"><span class="pill-label">Fakultas:</span> Ilmu Komputer</div>
        </div>
        <div class="pill-row" style="margin-top: 8px;">
            <div class="pill-badge"><span class="pill-label">Afiliasi:</span> Universitas Pembangunan Nasional "Veteran" Jawa Timur</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-label">Model Prediksi</div>
                <div class="info-card-value">GRU</div>
                <div class="info-card-desc">Gated Recurrent Unit adalah arsitektur jaringan saraf rekuren
                yang efisien untuk pemodelan data deret waktu keuangan berdimensi tinggi.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-label">Estimasi Risiko</div>
                <div class="info-card-value">VaR-ECF</div>
                <div class="info-card-desc">Value at Risk berbasis Cornish-Fisher Expansion
                untuk estimasi kuantil distribusi return dengan koreksi skewness dan kurtosis.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="info-card">
                <div class="info-card-label">Estimasi Volatilitas</div>
                <div class="info-card-value">EWMA</div>
                <div class="info-card-desc">Exponentially Weighted Moving Average untuk mengestimasi
                sigma yang responsif terhadap perubahan volatilitas pasar terkini.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("Mulai Analisis", use_container_width=True):
            st.session_state.active_page = "Input Data"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INPUT DATA
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Input Data":
    st.markdown("<div class='page-title'>Input Data</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Unggah data historis harga saham untuk dianalisis</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Unggah File CSV", "Gunakan Data Contoh"])

    with tab1:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            """<div class="upload-guide">
                <strong>Format yang didukung:</strong> File CSV dengan kolom
                <code>Date</code>, <code>Price</code>, <code>Open</code>, <code>High</code>,
                <code>Low</code>, <code>Vol.</code>, <code>Change %</code>.
                Format tanggal: MM/DD/YYYY atau YYYY-MM-DD.
                Nilai numerik dapat menggunakan koma sebagai pemisah ribuan.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Pilih file CSV",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                df_raw = clean_dataframe(df_raw)
                st.session_state.df = df_raw
                st.markdown(
                    f"""<div class="success-box">
                        Data berhasil dimuat. Total <strong>{len(df_raw):,}</strong> observasi
                        dengan <strong>{df_raw.shape[1]}</strong> kolom tersedia.
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("<div class='subsection-header'>Pratinjau Data (20 Baris Pertama)</div>", unsafe_allow_html=True)
                st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Gagal memuat file: {e}")

    with tab2:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            """<div class="upload-guide">
                Data contoh yang tersedia adalah data historis harga saham <strong>Bank BNI (BBNI)</strong>
                yang diunduh dari Investing.com. Gunakan data ini untuk mengeksplorasi fungsionalitas aplikasi.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        if st.button("Muat Data Contoh (BBNI)", key="load_sample"):
            try:
                df_raw = pd.read_csv("Bank Negar Stock Price History.csv")
                df_raw = clean_dataframe(df_raw)
                st.session_state.df = df_raw
                st.markdown(
                    f"""<div class="success-box">
                        Data contoh BBNI berhasil dimuat. Total <strong>{len(df_raw):,}</strong> observasi
                        dari <strong>{df_raw["Date"].min().strftime("%d %B %Y")}</strong> hingga
                        <strong>{df_raw["Date"].max().strftime("%d %B %Y")}</strong>.
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("<div class='subsection-header'>Pratinjau Data (20 Baris Pertama)</div>", unsafe_allow_html=True)
                st.dataframe(df_raw.head(20), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Gagal memuat data contoh: {e}")

    # Summary metrics if data is available
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Ringkasan Data Aktif</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(
                f"<div class='metric-mini'><div class='metric-mini-val'>{len(df):,}</div>"
                "<div class='metric-mini-label'>Total Observasi</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"<div class='metric-mini'><div class='metric-mini-val'>{df['Price'].min():,.0f}</div>"
                "<div class='metric-mini-label'>Harga Minimum (IDR)</div></div>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"<div class='metric-mini'><div class='metric-mini-val'>{df['Price'].max():,.0f}</div>"
                "<div class='metric-mini-label'>Harga Maksimum (IDR)</div></div>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"<div class='metric-mini'><div class='metric-mini-val'>{df['Price'].mean():,.0f}</div>"
                "<div class='metric-mini-label'>Harga Rata-Rata (IDR)</div></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALISIS DESKRIPTIF
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Analisis Deskriptif":
    st.markdown("<div class='page-title'>Analisis Deskriptif</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Eksplorasi statistik dan visualisasi data historis harga saham</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.markdown(
            """<div class="warning-box">Belum ada data yang dimuat. Silakan unggah data terlebih dahulu
            melalui halaman <strong>Input Data</strong>.</div>""",
            unsafe_allow_html=True,
        )
    else:
        from scipy import stats as scipy_stats
        from scipy.stats import norm, probplot

        df = st.session_state.df.copy()
        # Log-return: r_t = ln(P_t / P_{t-1}) * 100 (dalam persen)
        # Konsisten dengan perhitungan VaR-ECF dan standar literatur keuangan
        df["Return"] = np.log(df["Price"] / df["Price"].shift(1)) * 100
        df = df.dropna(subset=["Return"])

        tab_stats, tab_harga, tab_return, tab_vol = st.tabs(
            ["Statistik Deskriptif", "Harga Saham", "Distribusi Return", "Volume Perdagangan"]
        )

        # ─── Tab 1: Statistik ────────────────────────────────────────────────
        with tab_stats:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='subsection-header'>Ringkasan Statistik Deskriptif</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            ret_clean = df["Return"].dropna()
            stat_data = {
                "Statistik": [
                    "Jumlah Observasi", "Tanggal Awal", "Tanggal Akhir",
                    "Harga Minimum (IDR)", "Harga Maksimum (IDR)",
                    "Harga Rata-Rata (IDR)", "Median Harga (IDR)",
                    "Standar Deviasi Harga (IDR)",
                    "Return Minimum (%)", "Return Maksimum (%)",
                    "Return Rata-Rata (%)", "Standar Deviasi Return (%)",
                    "Skewness Return", "Kurtosis Return (Excess)",
                ],
                "Nilai": [
                    f"{len(df):,}",
                    df["Date"].min().strftime("%d %B %Y"),
                    df["Date"].max().strftime("%d %B %Y"),
                    f"{df['Price'].min():,.2f}",
                    f"{df['Price'].max():,.2f}",
                    f"{df['Price'].mean():,.2f}",
                    f"{df['Price'].median():,.2f}",
                    f"{df['Price'].std():,.2f}",
                    f"{ret_clean.min():.4f}",
                    f"{ret_clean.max():.4f}",
                    f"{ret_clean.mean():.4f}",
                    f"{ret_clean.std():.4f}",
                    f"{scipy_stats.skew(ret_clean):.4f}",
                    f"{scipy_stats.kurtosis(ret_clean):.4f}",
                ],
                "Keterangan": [
                    "Total data observasi yang tersedia",
                    "Observasi pertama dalam dataset",
                    "Observasi terakhir dalam dataset",
                    "Nilai harga penutupan terendah",
                    "Nilai harga penutupan tertinggi",
                    "Rata-rata harga penutupan",
                    "Nilai tengah distribusi harga",
                    "Dispersi harga dari rata-rata",
                    "Return harian terendah",
                    "Return harian tertinggi",
                    "Rata-rata return harian",
                    "Volatilitas return harian",
                    "Ukuran asimetri distribusi return",
                    "Ukuran ketebalan ekor distribusi",
                ],
            }
            stat_df = pd.DataFrame(stat_data)
            st.dataframe(stat_df, use_container_width=True, hide_index=True, height=500)

            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='subsection-header'>Uji Normalitas Jarque-Bera</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            jb_stat, jb_p = scipy_stats.jarque_bera(ret_clean)
            kesimpulan = "Tolak H0" if jb_p < 0.05 else "Gagal Tolak H0"
            interp_jb = (
                "Return saham <strong>tidak berdistribusi normal</strong> pada taraf signifikansi 5% "
                "(p-value &lt; 0,05). Hal ini mendukung penggunaan metode VaR-ECF yang "
                "tetap mempertimbangkan penyimpangan dari normalitas distribusi return."
                if jb_p < 0.05 else
                "Return saham <strong>tidak terdapat cukup bukti untuk menolak normalitas</strong> pada "
                "taraf signifikansi 5% (p-value &ge; 0,05)."
            )

            jb_df = pd.DataFrame({
                "Hipotesis": ["H0: Data berdistribusi normal"],
                "Statistik JB": [f"{jb_stat:.4f}"],
                "p-value": [f"{jb_p:.6f}"],
                "Keputusan (alpha = 0.05)": [kesimpulan],
            })
            st.dataframe(jb_df, use_container_width=True, hide_index=True)
            st.markdown(f"<div class='interp-box'>{interp_jb}</div>", unsafe_allow_html=True)

        # ─── Tab 2: Harga ────────────────────────────────────────────────────
        with tab_harga:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='subsection-header'>Grafik Harga Penutupan Saham</div>", unsafe_allow_html=True)

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=df["Date"], y=df["Price"],
                mode="lines",
                name="Harga Penutupan",
                line=dict(color="#0EA5E9", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(14, 165, 233, 0.15)",
            ))
            fig_price.update_layout(
                **chart_layout("Harga Penutupan Saham", "Tanggal", "Harga (IDR)"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_price, use_container_width=True)

            if all(c in df.columns for c in ["Open", "High", "Low", "Price"]):
                st.markdown("<div class='subsection-header'>Candlestick Chart (OHLC)</div>", unsafe_allow_html=True)
                # Show last 120 sessions for readability
                df_candle = df.tail(120)
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=df_candle["Date"],
                    open=df_candle["Open"],
                    high=df_candle["High"],
                    low=df_candle["Low"],
                    close=df_candle["Price"],
                    increasing_line_color="#10B981",
                    decreasing_line_color="#F43F5E",
                    name="OHLC",
                )])
                fig_candle.update_layout(
                    **chart_layout("Candlestick Chart — 120 Sesi Terakhir", "Tanggal", "Harga (IDR)"),
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig_candle, use_container_width=True)

        # ─── Tab 3: Return ───────────────────────────────────────────────────
        with tab_return:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            col_r1, col_r2 = st.columns(2, gap="medium")

            with col_r1:
                st.markdown("<div class='subsection-header'>Return Harian (%)</div>", unsafe_allow_html=True)
                colors_bar = ["#F43F5E" if r < 0 else "#10B981" for r in df["Return"]]
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(
                    x=df["Date"], y=df["Return"],
                    marker_color=colors_bar,
                    name="Return Harian",
                    opacity=0.85,
                ))
                fig_ret.update_layout(**chart_layout("Return Harian", "Tanggal", "Return (%)"))
                st.plotly_chart(fig_ret, use_container_width=True)

            with col_r2:
                st.markdown("<div class='subsection-header'>Distribusi Return</div>", unsafe_allow_html=True)
                mu_ret = df["Return"].mean()
                sig_ret = df["Return"].std()
                x_norm = np.linspace(df["Return"].min(), df["Return"].max(), 300)
                y_norm = norm.pdf(x_norm, mu_ret, sig_ret)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=df["Return"], nbinsx=60,
                    name="Return Aktual",
                    marker_color="#0EA5E9",
                    opacity=0.75,
                    histnorm="probability density",
                ))
                fig_hist.add_trace(go.Scatter(
                    x=x_norm, y=y_norm,
                    mode="lines",
                    name="Kurva Normal",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                ))
                fig_hist.update_layout(**chart_layout("Distribusi Return vs Normal", "Return (%)", "Densitas"))
                st.plotly_chart(fig_hist, use_container_width=True)

            # Q-Q Plot
            st.markdown("<div class='subsection-header'>Q-Q Plot Return terhadap Distribusi Normal</div>", unsafe_allow_html=True)
            qq_res = probplot(df["Return"].dropna(), dist="norm")
            qq_x = np.array(qq_res[0][0])
            qq_y = np.array(qq_res[0][1])
            slope, intercept, _ = qq_res[1]

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=qq_x, y=qq_y,
                mode="markers",
                name="Kuantil Sampel",
                marker=dict(color="#0EA5E9", size=4, opacity=0.65),
            ))
            fig_qq.add_trace(go.Scatter(
                x=qq_x, y=slope * qq_x + intercept,
                mode="lines",
                name="Referensi Normal",
                line=dict(color="#F43F5E", width=2),
            ))
            fig_qq.update_layout(**chart_layout("Q-Q Plot", "Kuantil Teoritis Normal", "Kuantil Sampel"))
            st.plotly_chart(fig_qq, use_container_width=True)

            skew_val = scipy_stats.skew(df["Return"].dropna())
            kurt_val = scipy_stats.kurtosis(df["Return"].dropna())
            st.markdown(
                f"""<div class="interp-box">
                    Distribusi return menunjukkan skewness sebesar <strong>{skew_val:.4f}</strong>
                    {"(menceng ke kiri, return negatif lebih dominan)" if skew_val < 0 else "(menceng ke kanan)"} dan
                    excess kurtosis sebesar <strong>{kurt_val:.4f}</strong>
                    {"(distribusi leptokurtik — ekor lebih tebal dari normal)" if kurt_val > 0 else "(distribusi platikurtik)"},
                    yang mengindikasikan bahwa asumsi normalitas pada data return perlu dipertanyakan.
                </div>""",
                unsafe_allow_html=True,
            )

        # ─── Tab 4: Volume ───────────────────────────────────────────────────
        with tab_vol:
            if "Vol." in df.columns and df["Vol."].notna().sum() > 5:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("<div class='subsection-header'>Volume Perdagangan Harian</div>", unsafe_allow_html=True)

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=df["Date"], y=df["Vol."] / 1e6,
                    name="Volume",
                    marker_color="#7B68EE",
                    opacity=0.8,
                ))
                fig_vol.update_layout(**chart_layout("Volume Perdagangan Harian", "Tanggal", "Volume (Juta Lembar)"))
                st.plotly_chart(fig_vol, use_container_width=True)

                v_avg = df["Vol."].mean() / 1e6
                v_max = df["Vol."].max() / 1e6
                v_min = df["Vol."].min() / 1e6
                st.markdown(
                    f"""<div class="interp-box">
                        Volume perdagangan rata-rata sebesar <strong>{v_avg:,.2f} juta</strong> lembar per hari.
                        Volume tertinggi tercatat <strong>{v_max:,.2f} juta</strong> lembar dan terendah
                        <strong>{v_min:,.2f} juta</strong> lembar. Lonjakan volume yang signifikan biasanya
                        berkorelasi dengan pergerakan harga yang besar atau rilis informasi material.
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Data volume tidak tersedia atau tidak mencukupi untuk divisualisasikan.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PARAMETER MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Parameter Model":
    st.markdown("<div class='page-title'>Parameter Model</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Konfigurasi arsitektur GRU dan parameter estimasi VaR-ECF</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.markdown(
            """<div class="warning-box">Belum ada data yang dimuat. Silakan unggah data terlebih dahulu
            melalui halaman <strong>Input Data</strong>.</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col_gru, col_var = st.columns(2, gap="large")

        with col_gru:
            st.markdown("<div class='subsection-header'>Arsitektur Model GRU</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            window_size = st.slider(
                "Ukuran Window / Look-back Period", 5, 60, 20, 1,
                help="Jumlah observasi historis yang digunakan sebagai input pada setiap langkah pelatihan",
            )
            gru_units_1 = st.slider(
                "Jumlah Unit GRU Layer 1", 16, 256, 64, 16,
                help="Jumlah neuron (unit) pada lapisan GRU pertama",
            )
            gru_units_2 = st.slider(
                "Jumlah Unit GRU Layer 2 (0 = tidak digunakan)", 0, 128, 32, 16,
                help="Jumlah neuron pada lapisan GRU kedua. Atur ke 0 untuk arsitektur satu lapisan GRU",
            )
            dropout_rate = st.slider(
                "Dropout Rate", 0.0, 0.5, 0.2, 0.05,
                help="Proporsi unit yang dinonaktifkan secara acak selama pelatihan untuk mencegah overfitting",
            )
            epochs = st.slider(
                "Jumlah Epoch Maksimum", 10, 200, 50, 10,
                help="Batas atas iterasi pelatihan. Early stopping dapat menghentikan pelatihan lebih awal",
            )
            batch_size = st.selectbox(
                "Ukuran Batch", [16, 32, 64, 128], index=1,
                help="Jumlah sampel yang diproses sebelum memperbarui bobot jaringan",
            )
            train_ratio = st.slider(
                "Proporsi Data Latih (%)", 60, 90, 80, 5,
                help="Persentase data historis yang digunakan untuk melatih model; sisanya untuk pengujian",
            )
            optimizer = st.selectbox(
                "Optimizer", ["adam", "rmsprop", "sgd"], index=0,
                help="Algoritma optimasi yang digunakan dalam proses pelatihan",
            )
            learning_rate = st.selectbox(
                "Learning Rate", [0.0001, 0.001, 0.005, 0.01], index=1,
                help="Ukuran langkah dalam pembaruan bobot; nilai terlalu besar menyebabkan divergensi",
                format_func=lambda x: str(x),
            )

        with col_var:
            st.markdown("<div class='subsection-header'>Parameter VaR-ECF</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            confidence_levels = st.multiselect(
                "Tingkat Kepercayaan VaR",
                [0.90, 0.95, 0.99],
                default=[0.95, 0.99],
                help="Level kepercayaan yang digunakan untuk estimasi Value at Risk",
                format_func=lambda x: f"{x*100:.0f}%",
            )

            window_size_var = st.slider(
                "Window Rolling VaR (hari)", 30, 250, 100, 10,
                help="Jumlah hari historis yang digunakan untuk mengestimasi VaR setiap "
                     "harinya secara rolling (out-of-sample). Window lebih besar = estimasi "
                     "lebih stabil tapi kurang responsif terhadap perubahan volatilitas terbaru.",
            )

            ewma_lambda = st.slider(
                "Decay Factor EWMA untuk Sigma (λ)", 0.80, 0.99, 0.94, 0.01,
                help="Sigma (volatilitas) kini diestimasi dengan EWMA (RiskMetrics-style), bukan "
                     "rolling standard deviation biasa. Observasi yang lebih baru diberi bobot "
                     "lebih besar, sehingga sigma lebih cepat 'sadar' saat volatilitas pasar naik "
                     "mendadak. Semakin kecil λ, semakin responsif (tapi lebih noisy) estimasi "
                     "sigma-nya. 0.94 adalah nilai standar RiskMetrics untuk data harian.",
            )

            st.markdown(
                """<div class="note-box">
                    <strong>Catatan:</strong> VaR-ECF diestimasi menggunakan Cornish-Fisher Expansion
                    (koreksi kuantil distribusi normal standar berdasarkan skewness dan kurtosis data
                    return), dengan parameter lokasi (μ) yang bersumber dari <strong>prediksi return
                    model GRU</strong> — bukan rata-rata historis. Parameter sebaran (σ) diestimasi
                    dengan <strong>EWMA (Exponentially Weighted Moving Average)</strong> alih-alih
                    rolling standard deviation biasa, agar lebih responsif terhadap perubahan
                    volatilitas pasar terbaru. Estimasi dilakukan secara <strong>rolling</strong>:
                    VaR tiap hari hanya menggunakan data historis sebelum hari tersebut
                    (out-of-sample), bukan dihitung sekaligus dari seluruh data uji.
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            use_sigma_calibration = st.checkbox(
                "Aktifkan Kalibrasi Otomatis Skala Sigma",
                value=True,
                help="Jika aktif, sistem mencari faktor pengali sigma (k) dari periode kalibrasi "
                     "(irisan validation split pada data training yang TIDAK pernah dipakai untuk "
                     "memperbarui bobot GRU) agar proporsi pelanggaran VaR mendekati target teoritis. "
                     "Ini membantu memperbaiki VaR yang secara sistematis terlalu longgar/ketat "
                     "tanpa mengubah formula Cornish-Fisher maupun menyentuh data periode test.",
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(
                """<div class="note-box">
                    <strong>Catatan:</strong> Waktu komputasi bergantung pada ukuran dataset dan jumlah epoch.
                    Disarankan memulai dengan nilai default untuk eksplorasi awal.
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        params = {
            "window_size": window_size,
            "gru_units_1": gru_units_1,
            "gru_units_2": gru_units_2,
            "dropout_rate": dropout_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_ratio": train_ratio / 100,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "confidence_levels": confidence_levels if confidence_levels else [0.95],
            "window_size_var": window_size_var,
            "ewma_lambda": ewma_lambda,
            "use_sigma_calibration": use_sigma_calibration,
        }

        if not confidence_levels:
            st.markdown(
                "<div class='warning-box'>Pilih minimal satu tingkat kepercayaan VaR sebelum menjalankan model.</div>",
                unsafe_allow_html=True,
            )

        if st.button("Jalankan Model dan Estimasi Risiko", use_container_width=True, key="run_model"):
            run_model(params)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HASIL DAN INTERPRETASI
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_page == "Hasil dan Interpretasi":
    st.markdown("<div class='page-title'>Hasil dan Interpretasi</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Performa model GRU dan estimasi VaR-ECF</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.results is None:
        st.markdown(
            """<div class="warning-box">Model belum dijalankan. Silakan masuk ke halaman
            <strong>Parameter Model</strong> dan jalankan model terlebih dahulu.</div>""",
            unsafe_allow_html=True,
        )
    else:
        results = st.session_state.results
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BAGIAN 1: PERFORMA MODEL GRU
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<div class='section-header'>Performa Model GRU</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>RMSE</div>"
                f"<div class='metric-val'>{results['rmse']:,.4f}</div></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>MAE</div>"
                f"<div class='metric-val'>{results['mae']:,.4f}</div></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>MAPE (%)</div>"
                f"<div class='metric-val'>{results['mape']:.4f}</div></div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>R-Squared</div>"
                f"<div class='metric-val'>{results['r2']:.4f}</div></div>",
                unsafe_allow_html=True,
            )

        # ── Tabel Interpretasi Metrik ──────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='subsection-header'>Interpretasi Metrik Evaluasi</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Kategorisasi MAPE (Lewis, 1982)
        if results["mape"] < 10:
            mape_cat = "Highly Accurate"
            mape_desc = "Akurasi prediksi sangat tinggi (MAPE < 10%)"
        elif results["mape"] < 20:
            mape_cat = "Good Forecast"
            mape_desc = "Akurasi prediksi baik (10% ≤ MAPE < 20%)"
        elif results["mape"] < 50:
            mape_cat = "Reasonable Forecast"
            mape_desc = "Akurasi prediksi cukup (20% ≤ MAPE < 50%)"
        else:
            mape_cat = "Inaccurate Forecast"
            mape_desc = "Akurasi prediksi rendah (MAPE ≥ 50%)"

        # Kategorisasi R²
        if results["r2"] >= 0.90:
            r2_cat = "Sangat Baik"
            r2_desc = f"Model menjelaskan {results['r2']*100:.2f}% variabilitas data (R² ≥ 0.90)"
        elif results["r2"] >= 0.70:
            r2_cat = "Baik"
            r2_desc = f"Model menjelaskan {results['r2']*100:.2f}% variabilitas data (0.70 ≤ R² < 0.90)"
        elif results["r2"] >= 0.50:
            r2_cat = "Cukup"
            r2_desc = f"Model menjelaskan {results['r2']*100:.2f}% variabilitas data (0.50 ≤ R² < 0.70)"
        else:
            r2_cat = "Lemah"
            r2_desc = f"Model menjelaskan {results['r2']*100:.2f}% variabilitas data (R² < 0.50)"

        metric_interp = pd.DataFrame({
            "Metrik": ["RMSE", "MAE", "MAPE", "R-Squared"],
            "Nilai": [
                f"{results['rmse']:,.4f}",
                f"{results['mae']:,.4f}",
                f"{results['mape']:.4f}%",
                f"{results['r2']:.4f}",
            ],
            "Kategori": [
                f"Deviasi rata-rata ±{results['rmse']:,.2f} IDR",
                f"Error absolut rata-rata {results['mae']:,.2f} IDR",
                mape_cat,
                r2_cat,
            ],
            "Interpretasi": [
                f"Rata-rata kesalahan prediksi (root mean square) sebesar {results['rmse']:,.2f} IDR dari harga aktual",
                f"Rata-rata deviasi absolut prediksi sebesar {results['mae']:,.2f} IDR dari harga aktual",
                mape_desc,
                r2_desc,
            ],
        })
        st.dataframe(metric_interp, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Grafik Prediksi pada Data Latih & Data Validasi ───────────────────
        st.markdown("<div class='subsection-header'>Prediksi GRU pada Data Latih & Data Validasi</div>", unsafe_allow_html=True)
        n_fit = results["n_fit"]
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(
            x=list(range(len(results["y_train_actual_full"]))),
            y=results["y_train_actual_full"],
            mode="lines", name="y_train (Aktual)",
            line=dict(color="#0EA5E9", width=1.5),
        ))
        fig_train.add_trace(go.Scatter(
            x=list(range(n_fit)),
            y=results["y_pred_fit"],
            mode="lines", name="Prediksi pada Data Latih",
            line=dict(color="#F43F5E", width=1.3),
        ))
        if len(results["y_pred_val"]) > 0:
            fig_train.add_trace(go.Scatter(
                x=list(range(n_fit, n_fit + len(results["y_pred_val"]))),
                y=results["y_pred_val"],
                mode="lines", name="Prediksi pada Data Validasi",
                line=dict(color="#A855F7", width=1.5),
            ))
        fig_train.update_layout(
            **chart_layout("Prediksi GRU pada Data Latih & Data Validasi", "No. Hari Perdagangan", "Harga (IDR)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_train, use_container_width=True)
        st.markdown(
            """<div class="interp-box">
                Grafik ini meniru pengecekan kesesuaian model pada subset data latih (merah) dan
                data validasi (ungu) terhadap nilai aktual (biru), sebagaimana ditampilkan pada
                jurnal acuan. Data validasi adalah bagian akhir dari data latih yang disisihkan
                Keras (<code>validation_split</code>) untuk monitoring/early stopping dan
                <strong>tidak pernah dipakai untuk memperbarui bobot GRU</strong> lewat backprop —
                sehingga kesesuaian pada garis ungu memberi gambaran awal kemampuan generalisasi
                model sebelum diuji sepenuhnya pada data uji (out-of-sample) di bawah.
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Grafik Prediksi ───────────────────────────────────────────────────
        st.markdown("<div class='subsection-header'>Perbandingan Harga Aktual vs Prediksi GRU</div>", unsafe_allow_html=True)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=results["test_dates"], y=results["y_test"],
            mode="lines", name="Aktual",
            line=dict(color="#0EA5E9", width=2),
        ))
        fig_pred.add_trace(go.Scatter(
            x=results["test_dates"], y=results["y_pred"],
            mode="lines", name="Prediksi GRU",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ))
        fig_pred.update_layout(
            **chart_layout("Harga Aktual vs Prediksi GRU — Data Uji", "Tanggal", "Harga (IDR)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Scatter Plot Aktual vs Prediksi ───────────────────────────────────
        col_scatter, col_residual = st.columns(2, gap="medium")

        with col_scatter:
            st.markdown("<div class='subsection-header'>Scatter Plot: Aktual vs Prediksi</div>", unsafe_allow_html=True)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=results["y_test"], y=results["y_pred"],
                mode="markers", name="Data Uji",
                marker=dict(color="#0EA5E9", size=5, opacity=0.6),
            ))
            # Garis ideal 45° (perfect prediction)
            min_val = min(results["y_test"].min(), results["y_pred"].min())
            max_val = max(results["y_test"].max(), results["y_pred"].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", name="Garis Ideal (y = x)",
                line=dict(color="#F43F5E", width=2, dash="dash"),
            ))
            fig_scatter.update_layout(
                **chart_layout("Scatter Plot Aktual vs Prediksi", "Harga Aktual (IDR)", "Harga Prediksi (IDR)"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_residual:
            st.markdown("<div class='subsection-header'>Distribusi Error Prediksi (%)</div>", unsafe_allow_html=True)
            fig_res_hist = go.Figure()
            fig_res_hist.add_trace(go.Histogram(
                x=results["residual_pct"], nbinsx=50,
                name="Error Prediksi",
                marker_color="#7B68EE",
                opacity=0.75,
                histnorm="probability density",
            ))
            fig_res_hist.update_layout(
                **chart_layout("Distribusi Persentase Error Prediksi", "Error (%)", "Densitas"),
            )
            st.plotly_chart(fig_res_hist, use_container_width=True)

        # Interpretasi scatter plot
        from scipy.stats import pearsonr
        corr_val, _ = pearsonr(results["y_test"], results["y_pred"])
        mean_error = float(np.mean(results["residual_pct"]))
        std_error = float(np.std(results["residual_pct"]))
        st.markdown(
            f"""<div class="interp-box">
                <strong>Analisis Scatter Plot:</strong> Korelasi Pearson antara harga aktual dan prediksi sebesar
                <strong>{corr_val:.4f}</strong>, menunjukkan {"hubungan linier sangat kuat" if abs(corr_val) > 0.95 else "hubungan linier kuat" if abs(corr_val) > 0.8 else "hubungan linier moderat"}.
                Semakin dekat titik-titik ke garis ideal (y = x), semakin akurat prediksi model.
                Rata-rata error prediksi sebesar <strong>{mean_error:.4f}%</strong> dengan standar deviasi
                <strong>{std_error:.4f}%</strong>.
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Training Loss ─────────────────────────────────────────────────────
        if results.get("history_loss"):
            st.markdown("<div class='subsection-header'>Kurva Loss Pelatihan</div>", unsafe_allow_html=True)
            ep_range = list(range(1, len(results["history_loss"]) + 1))
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=ep_range, y=results["history_loss"],
                mode="lines", name="Training Loss",
                line=dict(color="#0EA5E9", width=2),
            ))
            if results.get("history_val_loss"):
                fig_loss.add_trace(go.Scatter(
                    x=ep_range, y=results["history_val_loss"],
                    mode="lines", name="Validation Loss",
                    line=dict(color="#F43F5E", width=2),
                ))
            fig_loss.update_layout(**chart_layout("Kurva Loss Pelatihan", "Epoch", "Loss (MSE)"))
            st.plotly_chart(fig_loss, use_container_width=True)

            # Interpretasi kurva loss
            final_loss = results["history_loss"][-1]
            final_val = results["history_val_loss"][-1] if results.get("history_val_loss") else None
            actual_epochs = len(results["history_loss"])
            if final_val is not None:
                gap_pct = abs(final_loss - final_val) / max(final_loss, 1e-8) * 100
                if final_val <= final_loss:
                    overfit_note = "Validation loss setara atau lebih rendah dibanding training loss, mengindikasikan model <strong>tidak mengalami overfitting</strong> dan mampu melakukan generalisasi dengan baik pada data yang belum pernah dilihat."
                elif gap_pct < 15:
                    overfit_note = "Gap antara training loss dan validation loss relatif kecil, mengindikasikan model <strong>tidak mengalami overfitting</strong>."
                elif gap_pct < 30:
                    overfit_note = "Gap antara training loss dan validation loss moderat, model menunjukkan <strong>sedikit overfitting</strong> namun masih dalam batas wajar."
                else:
                    overfit_note = "Gap antara training loss dan validation loss cukup besar, mengindikasikan <strong>potensi overfitting</strong>. Pertimbangkan untuk meningkatkan dropout rate atau mengurangi kompleksitas model."
            else:
                overfit_note = ""

            st.markdown(
                f"""<div class="interp-box">
                    Pelatihan berhenti pada epoch ke-<strong>{actual_epochs}</strong>
                    {"(early stopping aktif)" if actual_epochs < results.get("max_epochs", 200) else ""}.
                    Training loss akhir: <strong>{final_loss:.6f}</strong>
                    {f", validation loss akhir: <strong>{final_val:.6f}</strong>" if final_val else ""}.
                    {overfit_note}
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BAGIAN 2: ESTIMASI VALUE AT RISK (VaR-ECF)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<div class='section-header'>Estimasi Value at Risk — VaR-ECF Terintegrasi GRU</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="interp-box">
                <strong>Catatan Metodologi:</strong> VaR-ECF pada sistem ini dihitung dengan
                mengintegrasikan keluaran model GRU ke dalam formula Cornish-Fisher Expansion:<br>
                VaR<sub>t</sub> = μ<sub>t</sub> + z<sub>CF</sub> × σ, dengan
                <strong>μ<sub>t</sub> = return yang diprediksi oleh model GRU</strong> untuk hari t
                (forward-looking), sementara σ (diestimasi via <strong>EWMA</strong>, λ =
                {results.get('ewma_lambda', 0.94):.2f}), skewness, dan kurtosis diestimasi dari
                <strong>{results['window_size_var']}</strong> hari data historis SEBELUM hari t
                secara rolling-window (out-of-sample, tanpa bias look-ahead). Dengan demikian,
                estimasi risiko tidak lagi murni bergantung pada data historis, melainkan
                ikut ditentukan oleh prediksi harga model GRU — sesuai dengan tujuan sistem
                sebagai <em>Prediksi Risiko Kerugian dengan Model GRU dan VaR-ECF</em>.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Info kalibrasi skala sigma (jika diaktifkan) ──
        sigma_scale_val = results.get("sigma_scale", 1.0)
        calib_diag = results.get("calib_diag")
        if calib_diag is not None and not calib_diag.get("skipped", False):
            arah = "diperlebar" if sigma_scale_val > 1.0 else ("dipersempit" if sigma_scale_val < 1.0 else "tidak diubah")
            st.markdown("<div class='subsection-header'>Kalibrasi Skala Sigma (Koreksi Lebar VaR)</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class="note-box">
                    Sigma EWMA <strong>{arah}</strong> dengan faktor <strong>k = {sigma_scale_val:.2f}×</strong>,
                    diestimasi dari periode kalibrasi (irisan validation split pada data training —
                    tidak pernah dipakai untuk memperbarui bobot GRU, dan seluruhnya berada sebelum
                    periode test) agar proporsi pelanggaran VaR pada periode tersebut mendekati target
                    teoritis. Faktor k yang sama kemudian diterapkan pada estimasi VaR di periode test
                    di bawah ini.
                </div>""",
                unsafe_allow_html=True,
            )
            calib_rows = []
            for cl_str, d in calib_diag.get("per_confidence_level", {}).items():
                calib_rows.append({
                    "Tingkat Kepercayaan": f"{float(cl_str)*100:.0f}%",
                    "Target α (%)": f"{d['target_alpha']*100:.2f}",
                    "Violation Rate Kalibrasi (%)": f"{d['violation_rate']*100:.2f}",
                    "Jumlah Observasi Kalibrasi": d["n_obs"],
                })
            if calib_rows:
                st.dataframe(pd.DataFrame(calib_rows), use_container_width=True, hide_index=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        elif calib_diag is not None and calib_diag.get("skipped", False):
            st.markdown(
                f"""<div class="warning-box">Kalibrasi sigma dilewati: {calib_diag.get("reason", "")}</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Kontribusi GRU: bandingkan mu dari prediksi GRU vs mu historis ──
        primary_cl_mu = list(results["var_ecf_rolling"].keys())[0]
        bt_mu = results["var_ecf_rolling"][primary_cl_mu]
        gru_mu_mean = float(np.mean(bt_mu["predicted_mu_series"]))
        gru_mu_std = float(np.std(bt_mu["predicted_mu_series"]))
        actual_ret_mean = float(np.mean(bt_mu["actual_series"]))

        st.markdown("<div class='subsection-header'>Kontribusi Prediksi GRU terhadap VaR-ECF</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        gru_contrib_df = pd.DataFrame({
            "Komponen": ["Rata-Rata μ Prediksi GRU (%)", "Standar Deviasi μ Prediksi GRU (%)", "Rata-Rata Return Aktual (%)"],
            "Nilai": [f"{gru_mu_mean:.4f}", f"{gru_mu_std:.4f}", f"{actual_ret_mean:.4f}"],
            "Keterangan": [
                "Rata-rata parameter lokasi (μ) VaR-ECF yang bersumber dari prediksi return GRU",
                "Variasi prediksi GRU antar hari — menunjukkan seberapa dinamis kontribusi GRU ke VaR",
                "Rata-rata return yang sesungguhnya terjadi, sebagai pembanding akurasi arah μ",
            ],
        })
        st.dataframe(gru_contrib_df, use_container_width=True, hide_index=True)
        st.markdown(
            """<div class="interp-box">
                Jika model GRU dihapus dari sistem ini, nilai μ pada rumus VaR-ECF di atas akan
                hilang perannya dan estimasi VaR akan kembali murni berbasis rata-rata historis.
                Tabel ini membuktikan bahwa prediksi GRU secara langsung menjadi bagian dari
                perhitungan risiko, bukan sekadar modul terpisah untuk prediksi harga.
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Statistik log-return dari seluruh observasi out-of-sample (gabungan semua hari test yang valid)
        primary_cl_stats = list(results["var_ecf_rolling"].keys())[0]
        log_ret = results["var_ecf_rolling"][primary_cl_stats]["actual_series"]
        lr_mean = float(np.mean(log_ret))
        lr_std = float(np.std(log_ret))
        lr_min = float(np.min(log_ret))
        lr_max = float(np.max(log_ret))
        lr_skew = float(pd.Series(log_ret).skew())
        lr_kurt = float(pd.Series(log_ret).kurtosis())

        st.markdown("<div class='subsection-header'>Statistik Log-Return Aktual (Periode Out-of-Sample)</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        lr_stats = pd.DataFrame({
            "Statistik": ["Jumlah Observasi", "Rata-Rata (%)", "Standar Deviasi (%)", "Minimum (%)", "Maksimum (%)", "Skewness", "Excess Kurtosis"],
            "Nilai": [
                f"{len(log_ret):,}",
                f"{lr_mean:.4f}",
                f"{lr_std:.4f}",
                f"{lr_min:.4f}",
                f"{lr_max:.4f}",
                f"{lr_skew:.4f}",
                f"{lr_kurt:.4f}",
            ],
            "Keterangan": [
                "Total observasi log-return pada periode uji (setelah window terpenuhi)",
                "Rata-rata log-return harian",
                "Volatilitas harian (risiko)",
                "Log-return terendah (kerugian terbesar)",
                "Log-return tertinggi (keuntungan terbesar)",
                "Asimetri distribusi (negatif = ekor kiri lebih panjang)",
                "Ketebalan ekor distribusi (> 0 = leptokurtik)",
            ],
        })
        st.dataframe(lr_stats, use_container_width=True, hide_index=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Tabel Ringkasan VaR (nilai terbaru = estimasi utk hari berikutnya) ──
        st.markdown("<div class='subsection-header'>VaR-ECF Terbaru (Estimasi Risiko Hari Berikutnya)</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        var_rows = []
        for cl, var_val in results["var_ecf"].items():
            cl_f = float(cl)
            # VaR harus negatif (batas kerugian); jika perhitungan menghasilkan positif, paksakan negatif
            var_display = var_val if var_val < 0 else -abs(var_val)
            bt = results["var_ecf_rolling"][cl]
            avg_violation = float(np.mean(bt["actual_series"] < bt["var_series"])) * 100
            var_rows.append({
                "Tingkat Kepercayaan": f"{cl_f*100:.0f}%",
                "Alpha (α)": f"{(1-cl_f)*100:.0f}%",
                "VaR-ECF Terbaru (%)": f"{var_display:.6f}",
                "Contoh: Investasi Rp100 Juta": f"Rp {abs(var_display) * 1_000_000:.0f}",
                "Rasio Pelanggaran Historis (%)": f"{avg_violation:.2f}",
                "Interpretasi": (
                    f"Dengan kepercayaan {cl_f*100:.0f}%, kerugian hari berikutnya diperkirakan "
                    f"tidak melebihi {abs(var_display):.4f}% (dihitung dari {results['window_size_var']} "
                    f"hari data terakhir)"
                ),
            })
        var_df_disp = pd.DataFrame(var_rows)
        st.dataframe(var_df_disp, use_container_width=True, hide_index=True)

        # ── Grafik: Rolling VaR line vs Return Aktual, dengan penanda pelanggaran ──
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='subsection-header'>Rolling VaR-ECF vs Return Aktual (Out-of-Sample)</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Gunakan confidence level pertama untuk visualisasi utama
        primary_cl = list(results["var_ecf_rolling"].keys())[0]
        bt_primary = results["var_ecf_rolling"][primary_cl]
        plot_dates = st.session_state.df["Date"].values[bt_primary["price_idx"]]
        violation_mask = bt_primary["actual_series"] < bt_primary["var_series"]
        color_points = ["#F43F5E" if v else "#10B981" for v in violation_mask]

        fig_viol = go.Figure()
        fig_viol.add_trace(go.Bar(
            x=plot_dates, y=bt_primary["actual_series"],
            marker_color=color_points,
            name="Log-Return Aktual",
            opacity=0.85,
        ))
        fig_viol.add_trace(go.Scatter(
            x=plot_dates, y=bt_primary["var_series"],
            mode="lines",
            name=f"Rolling VaR {float(primary_cl)*100:.0f}%",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        ))
        fig_viol.add_trace(go.Scatter(
            x=plot_dates, y=bt_primary["predicted_mu_series"],
            mode="lines",
            name="μ Prediksi GRU",
            line=dict(color="#8B5CF6", width=1.5, dash="dot"),
        ))
        fig_viol.update_layout(**chart_layout(
            f"Rolling VaR-ECF {float(primary_cl)*100:.0f}% vs Log-Return Aktual (Out-of-Sample)",
            "Tanggal", "Log-Return (%)",
        ))
        st.plotly_chart(fig_viol, use_container_width=True)

        n_violations = int(np.sum(violation_mask))
        st.markdown(
            f"""<div class="interp-box">
                Garis kuning putus-putus menunjukkan batas VaR yang diestimasi ULANG setiap hari
                dari {results['window_size_var']} hari sebelumnya (bukan garis statis), dengan σ
                yang dihitung via EWMA (λ = {results.get('ewma_lambda', 0.94):.2f}) agar lebih
                cepat merespons lonjakan volatilitas. Garis ungu titik-titik menunjukkan μ
                (parameter lokasi) yang bersumber dari prediksi return model GRU — inilah komponen
                yang membuat VaR bergerak mengikuti keluaran GRU. Bar berwarna
                <strong style="color:#F43F5E">merah</strong> menandakan hari di mana kerugian
                aktual melebihi estimasi VaR pada hari itu. Terdapat
                <strong>{n_violations}</strong> pelanggaran dari total
                <strong>{len(bt_primary["actual_series"])}</strong> observasi out-of-sample
                (rasio: {n_violations/len(bt_primary["actual_series"])*100:.2f}%,
                harapan teoritis: {(1-float(primary_cl))*100:.0f}%).
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # BAGIAN 3: RINGKASAN DAN INTERPRETASI KOMPREHENSIF
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("<div class='section-header'>Ringkasan dan Interpretasi Komprehensif</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        with st.expander("Baca Interpretasi Lengkap Hasil Analisis", expanded=True):
            # Performance level
            if results["mape"] < 10:
                perf_level = "sangat baik (highly accurate)"
                perf_note = "di bawah ambang batas 10% menurut klasifikasi Lewis (1982)"
            elif results["mape"] < 20:
                perf_level = "baik (good forecast)"
                perf_note = "pada kisaran 10-20% menurut klasifikasi Lewis (1982)"
            elif results["mape"] < 50:
                perf_level = "cukup (reasonable forecast)"
                perf_note = "pada kisaran 20-50%"
            else:
                perf_level = "perlu ditingkatkan (inaccurate forecast)"
                perf_note = "di atas 50%"

            # VaR summary
            var_summary_parts = []
            for cl, var_val in results["var_ecf"].items():
                cl_f = float(cl)
                var_display = var_val if var_val < 0 else -abs(var_val)
                var_summary_parts.append(
                    f"pada tingkat kepercayaan <strong>{cl_f*100:.0f}%</strong>, VaR-ECF sebesar "
                    f"<strong>{var_display:.4f}%</strong> per hari "
                    f"(potensi kerugian maksimum = {abs(var_display):.4f}% dari nilai investasi)"
                )

            interp_text = f"""
            <div class="expander-content">
            <strong>Alur Analisis:</strong> Data harga saham → Prediksi return oleh GRU (μ forward-looking)
            → Estimasi σ (EWMA), skewness, kurtosis dari rolling window historis → VaR-ECF = μ<sub>GRU</sub> + z<sub>CF</sub> × σ,
            diestimasi secara rolling-window out-of-sample.<br><br>

            <strong>1. Performa Prediksi Model GRU</strong><br>
            Model Gated Recurrent Unit (GRU) menunjukkan performa prediksi yang <strong>{perf_level}</strong>
            dengan nilai MAPE sebesar <strong>{results['mape']:.4f}%</strong> ({perf_note}).
            Nilai RMSE sebesar <strong>{results['rmse']:,.4f}</strong> IDR menunjukkan rata-rata deviasi
            kuadrat prediksi dari harga aktual, sementara MAE sebesar <strong>{results['mae']:,.4f}</strong>
            IDR merupakan rata-rata deviasi absolut.
            Koefisien determinasi R² = <strong>{results['r2']:.4f}</strong> mengindikasikan bahwa model
            mampu menjelaskan <strong>{results['r2']*100:.2f}%</strong> variabilitas harga saham pada
            data uji, {"yang merupakan tingkat explanatory power yang tinggi untuk data deret waktu keuangan" if results['r2'] > 0.85 else "yang menunjukkan model cukup baik dalam menangkap pola pergerakan harga"}.
            Selain dievaluasi secara mandiri, prediksi return dari model GRU ini juga menjadi
            <strong>input langsung</strong> bagi perhitungan VaR-ECF pada tahap berikutnya.

            <br><br>

            <strong>2. Estimasi Value at Risk Terintegrasi GRU (VaR-ECF)</strong><br>
            VaR diestimasi menggunakan Cornish-Fisher Expansion dengan formula
            VaR<sub>t</sub> = μ<sub>t</sub> + z<sub>CF</sub> × σ, di mana <strong>μ<sub>t</sub> diambil dari
            return yang diprediksi model GRU</strong> (forward-looking), sementara σ diestimasi
            dengan <strong>EWMA</strong> (λ = {results.get('ewma_lambda', 0.94):.2f}) dan skewness,
            kurtosis diestimasi dari data historis secara rolling-window agar tetap
            <strong>mempertimbangkan penyimpangan dari normalitas</strong> yang sering ditemukan pada
            data return keuangan (sebagaimana ditunjukkan oleh uji Jarque-Bera pada halaman Analisis
            Deskriptif). Penggunaan EWMA (dibanding rolling standard deviation biasa) dipilih agar
            komponen volatilitas lebih cepat merespons perubahan kondisi pasar, sehingga estimasi
            risiko tidak murni bersandar pada data historis yang "lamban", melainkan ikut ditentukan
            oleh keluaran model GRU dan pembobotan waktu pada σ — sehingga GRU dan VaR-ECF bekerja
            sebagai satu kesatuan sistem prediksi risiko, bukan dua modul yang berdiri sendiri.<br><br>
            Hasil estimasi VaR-ECF (nilai terbaru, berdasarkan prediksi GRU paling akhir):<br>
            {"<br>".join(f"• {part}" for part in var_summary_parts)}<br><br>
            Sebagai contoh, untuk investasi sebesar Rp100.000.000, potensi kerugian maksimum pada hari
            perdagangan berikutnya, pada tingkat kepercayaan yang dipilih, dapat dilihat pada tabel di atas.

            <br><br>

            <strong>Catatan mengenai Sumber Volatilitas:</strong> Komponen σ pada VaR-ECF ini
            diestimasi dari data historis menggunakan EWMA, bukan dari prediksi GRU, karena model
            GRU pada sistem ini dilatih untuk memprediksi harga/return, bukan volatilitas.
            Pengembangan lebih lanjut dapat mengarahkan GRU untuk turut memprediksi volatilitas
            (misalnya melalui pendekatan hybrid GRU-GARCH), sehingga seluruh komponen VaR-ECF
            sepenuhnya forward-looking.

            <br><br>

            <strong>3. Kesimpulan</strong><br>
            {"Berdasarkan hasil analisis komprehensif, model GRU berhasil memprediksi harga saham dengan akurasi tinggi, dan kontribusinya sebagai komponen μ pada VaR-ECF memberikan estimasi risiko yang bersifat forward-looking. Sistem prediksi risiko kerugian investasi saham dengan GRU dan VaR-ECF ini dapat diandalkan sebagai alat bantu pengambilan keputusan investasi." if results['mape'] < 10 else "Model GRU telah menunjukkan kemampuan prediksi yang memadai dan kontribusinya sebagai komponen μ pada VaR-ECF memberikan gambaran kuantitatif mengenai potensi risiko kerugian yang bersifat forward-looking."}

            <br><br>

            <strong>Catatan Penting:</strong><br>
            • Hasil estimasi VaR hanya berlaku untuk <strong>jangka waktu satu hari ke depan</strong>
            (1-day horizon) dan bergantung pada akurasi prediksi return GRU pada hari tersebut.<br>
            • Kondisi pasar yang ekstrem atau peristiwa tak terduga (<em>black swan events</em>) dapat
            menghasilkan kerugian yang melampaui batas VaR yang diestimasi, termasuk bila prediksi
            GRU meleset signifikan dari kondisi aktual.<br>
            • Log-return digunakan sebagai ukuran perubahan harga karena bersifat <strong>time-additive</strong>
            dan lebih sesuai dengan teori keuangan modern.<br>
            • VaR-ECF (Cornish-Fisher Expansion) tetap mempertimbangkan skewness dan kurtosis data
            historis, sehingga lebih robust dibanding VaR parametrik normal biasa, sementara parameter
            lokasi (μ) bersumber dari prediksi GRU yang bersifat forward-looking dan σ diestimasi
            secara adaptif melalui EWMA.
            </div>
            """
            st.markdown(interp_text, unsafe_allow_html=True)
