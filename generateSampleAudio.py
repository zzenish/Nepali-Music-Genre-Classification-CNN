import os
from subprocess import call
from concurrent.futures import ThreadPoolExecutor, as_completed

categories = {
    "POP": [
        "https://youtu.be/h8kqyqrFhdE",
        "https://youtu.be/hoWdmPZ35m0",
        "https://youtu.be/SEz2cpDC-Jc",
        "https://youtu.be/rFV1WHHxBCk",
    ],
    "Nephop": [
        "https://youtu.be/0WNvapwEIsQ",
        "https://youtu.be/AnczJYWQsGI",
        "https://youtu.be/L_zazwOSmPM",
        "https://youtu.be/aEQVvNIolrs",
        "https://youtu.be/zPor55p5SXs",
    ],
    "Gazal": [
        "https://youtu.be/yAqUmCcfSKI",
        "https://youtu.be/0XoQgKzE3w0",
        "https://youtu.be/dSeGiFhIaeA",
        "https://youtu.be/BYCZAnpG_x0",
        "https://youtu.be/EKOvB8Su4s8",
    ],
    "Lokdohori": [
        "https://youtu.be/04tF8V0cba4",
        "https://youtu.be/4HOtcYijiFc",
        "https://youtu.be/BbB2dNsjIDM",
        "https://youtu.be/NXmzi4SSKXA",
        "https://youtu.be/WgTCIn-pXHg",
    ]
}

base_folder = "testSample"

def download_audio(category, url):
    output_folder = os.path.join(base_folder, category)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Downloading {url} into {output_folder}")
    return call([
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", os.path.join(output_folder, "%(id)s.%(ext)s"),
        url
    ])

# Adjust max_workers depending on your CPU/network capacity; here I'm using 4 for demonstration.
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for category, urls in categories.items():
        for url in urls:
            futures.append(executor.submit(download_audio, category, url))

    for future in as_completed(futures):
        try:
            result = future.result()
            if result == 0:
                print("Download completed successfully")
            else:
                print("Download failed with code", result)
        except Exception as e:
            print("❌ Error:", e)
