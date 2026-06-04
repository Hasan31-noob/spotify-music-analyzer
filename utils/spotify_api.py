"""
spotify_api.py
==============
Lapisan tipis (thin wrapper) di atas Spotify Web API.

Modul ini menangani:
- Autentikasi Client Credentials (tanpa login user) menggunakan spotipy.
- Pencarian lagu, artis, dan album.
- Pengambilan detail lagu & artis (termasuk genre + popularity).

Catatan penting (sejak 27 Nov 2024):
Spotify men-deprecate endpoint Audio Features / Audio Analysis /
Recommendations untuk aplikasi BARU. Jadi modul ini sengaja TIDAK
memanggil endpoint tersebut — fitur audio diambil dari dataset lokal
(lihat utils/data_processing.py).
"""

from __future__ import annotations

import os
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ---------------------------------------------------------------------------
# Autentikasi
# ---------------------------------------------------------------------------
def _read_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Membaca Client ID & Client Secret secara aman.

    Urutan prioritas:
    1. st.secrets  (saat dideploy ke Streamlit Cloud)
    2. Environment variable / file .env  (saat dijalankan lokal)

    Client Secret TIDAK PERNAH ditulis langsung di dalam kode.
    """
    client_id = None
    client_secret = None

    # 1) Coba st.secrets terlebih dahulu (aman untuk deployment).
    try:
        import streamlit as st

        client_id = st.secrets.get("SPOTIFY_CLIENT_ID")
        client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET")
    except Exception:
        # st.secrets tidak tersedia / belum dikonfigurasi -> abaikan.
        pass

    # 2) Fallback ke environment variable (mis. dari file .env).
    if not client_id:
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
    if not client_secret:
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    return client_id, client_secret


def get_spotify_client() -> Optional[spotipy.Spotify]:
    """
    Membuat dan mengembalikan objek client Spotify yang sudah terautentikasi.

    Mengembalikan None apabila kredensial belum diisi, sehingga UI bisa
    menampilkan pesan yang ramah alih-alih error mentah.
    """
    client_id, client_secret = _read_credentials()
    if not client_id or not client_secret:
        return None

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10)


def credentials_available() -> bool:
    """Cek cepat apakah kredensial sudah tersedia."""
    cid, secret = _read_credentials()
    return bool(cid and secret)


# ---------------------------------------------------------------------------
# Helper format
# ---------------------------------------------------------------------------
def format_duration(duration_ms: int) -> str:
    """Mengubah durasi milidetik menjadi format mm:ss."""
    if not duration_ms:
        return "0:00"
    seconds = int(duration_ms) // 1000
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _best_image(images: list) -> Optional[str]:
    """Mengambil URL gambar pertama (resolusi tertinggi) jika ada."""
    if images:
        return images[0].get("url")
    return None


# ---------------------------------------------------------------------------
# Pencarian
# ---------------------------------------------------------------------------
def search_tracks(sp: spotipy.Spotify, query: str, limit: int = 10) -> list[dict]:
    """Mencari lagu berdasarkan kata kunci dan mengembalikan list yang rapi."""
    results = sp.search(q=query, type="track", limit=limit)
    items = results.get("tracks", {}).get("items", [])

    tracks = []
    for item in items:
        album = item.get("album", {})
        artists = item.get("artists", [])
        tracks.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "artist": ", ".join(a.get("name", "") for a in artists),
                "artist_ids": [a.get("id") for a in artists],
                "album": album.get("name"),
                "release_date": album.get("release_date"),
                "popularity": item.get("popularity", 0),
                "duration": format_duration(item.get("duration_ms", 0)),
                "duration_ms": item.get("duration_ms", 0),
                "explicit": item.get("explicit", False),
                "image": _best_image(album.get("images", [])),
                "preview_url": item.get("preview_url"),
                "spotify_url": item.get("external_urls", {}).get("spotify"),
            }
        )
    return tracks


def search_artists(sp: spotipy.Spotify, query: str, limit: int = 10) -> list[dict]:
    """Mencari artis berdasarkan kata kunci."""
    results = sp.search(q=query, type="artist", limit=limit)
    items = results.get("artists", {}).get("items", [])

    artists = []
    for item in items:
        artists.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "genres": item.get("genres", []),
                "popularity": item.get("popularity", 0),
                "followers": item.get("followers", {}).get("total", 0),
                "image": _best_image(item.get("images", [])),
                "spotify_url": item.get("external_urls", {}).get("spotify"),
            }
        )
    return artists


def search_albums(sp: spotipy.Spotify, query: str, limit: int = 10) -> list[dict]:
    """Mencari album berdasarkan kata kunci."""
    results = sp.search(q=query, type="album", limit=limit)
    items = results.get("albums", {}).get("items", [])

    albums = []
    for item in items:
        artists = item.get("artists", [])
        albums.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "artist": ", ".join(a.get("name", "") for a in artists),
                "release_date": item.get("release_date"),
                "total_tracks": item.get("total_tracks", 0),
                "image": _best_image(item.get("images", [])),
                "spotify_url": item.get("external_urls", {}).get("spotify"),
            }
        )
    return albums


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------
def get_artist_details(sp: spotipy.Spotify, artist_id: str) -> dict:
    """Mengambil detail lengkap seorang artis (termasuk genre)."""
    item = sp.artist(artist_id)
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "genres": item.get("genres", []),
        "popularity": item.get("popularity", 0),
        "followers": item.get("followers", {}).get("total", 0),
        "image": _best_image(item.get("images", [])),
        "spotify_url": item.get("external_urls", {}).get("spotify"),
    }


def get_artist_top_tracks(
    sp: spotipy.Spotify, artist_id: str, market: str = "US", limit: int = 5
) -> list[dict]:
    """Mengambil lagu-lagu terpopuler dari seorang artis."""
    results = sp.artist_top_tracks(artist_id, country=market)
    items = results.get("tracks", [])[:limit]

    tracks = []
    for item in items:
        album = item.get("album", {})
        tracks.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "album": album.get("name"),
                "popularity": item.get("popularity", 0),
                "duration": format_duration(item.get("duration_ms", 0)),
                "image": _best_image(album.get("images", [])),
                "spotify_url": item.get("external_urls", {}).get("spotify"),
            }
        )
    return tracks


def get_track_genres(sp: spotipy.Spotify, artist_ids: list[str]) -> list[str]:
    """
    Karena objek track tidak menyimpan genre, genre diambil dari artis.
    Mengembalikan daftar genre unik dari semua artis lagu tersebut.
    """
    genres: list[str] = []
    for aid in artist_ids:
        if not aid:
            continue
        try:
            artist = sp.artist(aid)
            genres.extend(artist.get("genres", []))
        except Exception:
            continue
    # Hilangkan duplikat sambil mempertahankan urutan.
    return list(dict.fromkeys(genres))
