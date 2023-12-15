from pytube import YouTube
import argparse
import moviepy.editor as mp 
import speech_recognition as sr
import whisper_timestamped as whisper
from pydub import AudioSegment
import librosa 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import IPython.display as ipd 
import os,shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import concurrent.futures
import subprocess


name = ""
def download_youtube_video(url):
	global name
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
		name = yt.title

	except Exception as e:
		print(f"Download error: {e}")

parser = argparse.ArgumentParser(description = "YouTube Video Download")
parser.add_argument("url", help = "Video URL")

args = parser.parse_args()

download_youtube_video(args.url)

# Replace 'your_video_path' with the actual path to the video file

def get_audio():
	VideoFileClip("output.mp4").audio.write_audiofile("output.mp3")
 
	# audio = AudioSegment.from_wav("output.wav")
	'''
	# Export as MP3
	audio.export("output.mp3", format="mp3")
	'''
get_audio()


def get_transcription():
	audio = whisper.load_audio('output.wav')
	model = whisper.load_model("base")

	result = whisper.transcribe(model, audio, language="en")
	dictionary = {}
	for i, segment in enumerate(result['segments']):
		start, end = segment['start'], segment['end']
		dictionary[segment['text'].strip()] = str(int(start))
	return dictionary


#Load the audio file of the sports clip.
filename='output.mp3' #Enter your audio file name of match here. .wav,.mp3, etc. are supported.
vid, sample_rate = librosa.load(filename,sr=16000)
print(int(librosa.get_duration(y = vid, sr = sample_rate)/60))

#Breaking down video into chunks of 5 seconds so that rise in energy can be found.
chunk_size=5 
window_length = chunk_size * sample_rate

#seeing an audio sample and it's time-amplitude graph
a=vid[5*window_length:6*window_length] 
ipd.Audio(a, rate=sample_rate)
energy = sum(abs(a**2))
print(energy)

fig = plt.figure(figsize=(14, 8)) 
ax1 = fig.add_subplot(211) 
ax1.set_xlabel('Time') 
ax1.set_ylabel('Amplitude') 
ax1.plot(a)


#Plotting short time energy distribution histogram of all chunks
energy = np.array([sum(abs(vid[i:i+window_length]**2)) for i in range(0, len(vid), window_length)])
plt.hist(energy) 
plt.show()
#Close graphs for progress of program

#Finding and setting threshold value of commentator and audience noise above which we want to include portion in highlights.
df=pd.DataFrame(columns=['energy','start','end'])
thresh=300
row_index=0
for i in range(len(energy)):
	value=energy[i]
	if(value>=thresh):
		i=np.where(energy == value)[0]
		df.loc[row_index,'energy']=value
		df.loc[row_index,'start']=i[0] * 5
		df.loc[row_index,'end']=(i[0]+1) * 5
		row_index= row_index + 1

#Merge consecutive time intervals of audio clips into one.
temp=[]
i,j,n=0,0,len(df) - 1
while(i<n):
	j=i+1
	while(j<=n):
		if(df['end'][i] == df['start'][j]):
			df.loc[i,'end'] = df.loc[j,'end']
			temp.append(j)
			j=j+1
		else:
			i=j
			break  
df.drop(temp,axis=0,inplace=True)


#Extracting subclips from the video file on the basis of energy profile obtained from audio file.
start=np.array(df['start'])
end=np.array(df['end'])

#Create temporary folder for storing subclips generated. This folder will be deleted later after highlights are generated. 
cwd=os.getcwd()
sub_folder=os.path.join(cwd,"Subclips")
if os.path.exists(sub_folder):
	shutil.rmtree(sub_folder)
	path=os.mkdir(sub_folder)
else:
	path=os.mkdir(sub_folder)
#print(sub_folder,type(sub_folder))

#Extract moments from videos to be added in highlight
print(df)
begin = {}
stop = {}
filenames = {}

def process(start_lim, end_lim, filename):
	with VideoFileClip("output.mp4") as video:
		new = video.subclip(start_lim, end_lim)
		new.write_videofile(sub_folder+"/"+filename, audio_codec='aac')

for i in range(len(df)):
	if(i>=5):
		begin[i] = start[i] - 5  #Assuming that noise starts after the shot, so set start point as t-5 seconds to include the shot/wicket action.
	else:
		begin[i] = start[i] 
	stop[i] = end[i] + 2  
	filenames[i] = "highlight" + str(i+1) + ".mp4"
	
with concurrent.futures.ThreadPoolExecutor() as executor:
	list(executor.map(process, begin.values(), stop.values(), filenames.values()))



files=os.listdir(sub_folder)
files=[sub_folder+"/highlight" + str(i+1) + ".mp4" for i in range(len(df))]
#print(files)
final_clip=concatenate_videoclips([VideoFileClip(i) for i in files])
final_clip.write_videofile("highlight.mp4") #Enter the desired output highlights filename.
shutil.rmtree(sub_folder) #Delete the temporary file.
os.remove("output.mp4")
os.remove("output.mp3")
#os.remove("output.wav")
