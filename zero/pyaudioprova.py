import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "recordedFile.wav"
device_index = 2
audio = pyaudio.PyAudio()

print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")

index = int(input())
print("recording via index "+str(index))

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = index,
                frames_per_buffer=CHUNK)
print ("recording started")
Recordframes = []
Recordbytes = b''
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
#    print(int.from_bytes(data,'little'))
#    Recordframes.append(data)
    Recordframes.append(int.from_bytes(data,'big',signed=True))

print ("recording stopped")

print("conversion started")
for i in Recordframes:
    Recordbytes += i.to_bytes(audio.get_sample_size(FORMAT),'big',signed=True)
print("conversion finished")

print(len(Recordframes))
print(len(Recordbytes))

stream.stop_stream()
stream.close()
audio.terminate()

outf=open("record.txt","w")
for item in Recordframes:
        outf.write("%d\n" % item)
outf.close()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(Recordbytes)
waveFile.close()