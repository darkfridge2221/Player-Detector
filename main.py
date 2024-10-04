import tkinter as tk
from tkinter import filedialog, messagebox
from pytubefix import YouTube
import os
from analyze import analyze_video  #Import the analyse function

#Function to download video from YouTube
def download_youtube_video(link):
    try:
        yt = YouTube(link)
        video = yt.streams.get_highest_resolution()
        video_path = video.download(output_path=".")
        analyze_video(video_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download video: {e}")

#Function to open file dialog and select video file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        analyze_video(file_path)

#Function to handle the download button
def download_video():
    link = entry.get()
    if link:
        download_youtube_video(link)
    else:
        messagebox.showwarning("Input Error", "Please enter a valid YouTube link.")

#Main application window
root = tk.Tk()
root.title("Volleyball Video Analyzer")

#Frame for file selection
frame = tk.Frame(root)
frame.pack(pady=20)

btn_select = tk.Button(frame, text="Select Video File", command=select_file)
btn_select.pack(side=tk.LEFT, padx=10)

label = tk.Label(frame, text="Or enter YouTube link:")
label.pack(side=tk.LEFT)

entry = tk.Entry(frame, width=30)
entry.pack(side=tk.LEFT, padx=10)

btn_download = tk.Button(frame, text="Download Video", command=download_video)
btn_download.pack(side=tk.LEFT, padx=10)

root.mainloop()
