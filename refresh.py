import os
import glob

def hapus_foto_di_folder(folder_path, ekstensi=['.jpg', '.jpeg', '.png']):
    try:
        patterns = [os.path.join(folder_path, '*' + ext) for ext in ekstensi]
        files_to_delete = []
        for pattern in patterns:
            files_to_delete.extend(glob.glob(pattern))
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"File {file_path} berhasil dihapus.")

        print("Semua foto berhasil dihapus.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
folder_static = 'static/'
hapus_foto_di_folder(folder_static)
