from pytube import YouTube
import argparse
import moviepy.editor as mp 
import whisper_timestamped as whisper
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from pyannote.audio import Pipeline
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook

torch.cuda.set_device(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def download_youtube_video(url):
	try:
		yt = YouTube(url)
		'''
		video_stream = yt.streams.filter(res = "1080p").first()

		video_stream.download(".", filename = "output.mp4")

		audio_stream = yt.streams.filter(only_audio = True).first()

		audio_stream.download(".", filename = "output.webm")
		# subprocess.run('ffmpeg -i "output".webm "output".mp3',shell=True)

		video_clip = VideoFileClip("output.mp4")
		audio_clip = AudioFileClip("output.webm")
		print("fps")
		print(audio_clip.fps)
		print(video_clip.fps)
		audio_clip.set_fps(video_clip.fps)
		video_clip = video_clip.set_audio(audio_clip)

		video_clip.write_videofile("output.mp4", codec="libx264", audio_codec="aac")
		'''
		video_stream = yt.streams.get_highest_resolution()
		video_stream.download(".", filename = "output.mp4")
		print(f"Download completed: {yt.title}")

	except Exception as e:
		print(f"Download error: {e}")

parser = argparse.ArgumentParser(description = "YouTube Video Download")
parser.add_argument("url", help = "Video URL")

args = parser.parse_args()

download_youtube_video(args.url)

# Replace 'your_video_path' with the actual path to the video file

def get_audio():
	video = VideoFileClip("output.mp4")
	print(video.duration)
	video.audio.write_audiofile("output.mp3")
	print(video.duration)
	print("downloaded")
get_audio()

def get_diarization():

	pipeline = Pipeline.from_pretrained(
		"pyannote/speaker-diarization-3.1",
		use_auth_token="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")
	pipeline = pipeline.to(torch.device('cuda'))
	print('pipeline intialized')

	with ProgressHook() as hook:
		diarization = pipeline("output.mp3", hook = hook)
	print('diarization finished')

	diarization_person = []
	diarization_times = []
	for segment, thing, speaker in diarization.itertracks(yield_label=True):
		if diarization_person != []:
			if diarization_person[-1] == str(speaker):
				diarization_times[-1][1] = segment.end
			else:
				diarization_person.append(str(speaker))
				diarization_times.append([segment.start, segment.end])
		else:
			diarization_person.append(str(speaker))
			diarization_times.append([segment.start, segment.end])	
	return (diarization_person, diarization_times)

diarization = get_diarization()
print(diarization)

def get_transcription():
	audio = whisper.load_audio('output.mp3')
	model = whisper.load_model("medium")

	result = whisper.transcribe(model, audio, language="en")
	dictionary = {}
	with open("transcription_timestamps.txt", 'w') as timestamp_file:
		with open("transcription.txt", 'w') as file:
			for i, segment in enumerate(result['segments']):
				start, end = segment['start'], segment['end']
				dictionary[segment['text'].strip()] = str(start)
				timestamp_file.write(str(start) + " : " + segment['text'].strip() +"\n")
				file.write(segment['text'].strip() + " ")
	return dictionary

transcription = get_transcription()
print(transcription)

def compose_transcription(transcription, diarization):
	final_transcription = transcription
	for i in range(len(diarization[0])):
		for j in transcription.keys():
			if float(transcription[j]) <= diarization[1][i][1]:
				final_transcription[j] = [transcription[j], diarization[0][i]]
	return final_transcription

print(compose_transcription(transcription, diarization))
