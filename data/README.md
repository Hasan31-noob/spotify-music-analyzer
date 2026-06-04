# Folder Data

Letakkan dataset Spotify Anda di sini dengan nama **`data.csv`**.

Dataset harus memiliki kolom audio features berikut beserta kolom target `liked`:

```
danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, duration_ms, time_signature, liked
```

`liked` adalah label biner: `1` = lagu disukai, `0` = tidak disukai.

> Jika `data.csv` tidak ditemukan, aplikasi otomatis memakai **dataset demo
> sintetis** agar tetap bisa dijalankan untuk keperluan demonstrasi.
