# n8n Production Setup

Repository ini berisi konfigurasi n8n production automation yang sudah dioptimasi untuk berjalan di server dengan resource terbatas (RAM 1GB).

## Fitur & Optimasi
- **Memory Management**: Konfigurasi `NODE_OPTIONS` untuk membatasi heap size agar tidak crash OOM.
- **Persistent Storage**: Fix sinkronisasi folder session WhatsApp (Gowa) agar tidak perlu scan ulang.
- **Auto Backup**: Script otomatis backup workflows & credentials setiap 2 jam ke GitHub.
- **Cloudflare Tunnel**: Terintegrasi dengan Cloudflare untuk akses aman tanpa buka port.

## Struktur Folder
- `docker-compose.yml`: Main orchestration.
- `workflows/`: Export terbaru dari workflows dan credentials.
- `scripts/`: Utility scripts untuk backup dan sync.

## Cara Menggunakan Backup
Untuk restore workflows dan credentials ke server baru:
1. Jalankan `docker compose up -d`.
2. Gunakan script sync: `./scripts/sync-workflows.sh import-all`.

---
*Last Updated: 2026-01-20*
