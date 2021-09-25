from __future__ import print_function, division
import random
import quotes
import face_recognition
import cv2
import numpy as np
from playsound import playsound
from google.cloud import texttospeech
from time import gmtime, strftime
from selenium import webdriver
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue
import sqlite3 as sl
import board, pieces, ai
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from playsound import playsound
from google.cloud import texttospeech

#grabbing quotes from quotes list
ranQuote = random.choice(quotes.quotesList)

#opening camera for computer vision
video_capture = cv2.VideoCapture(0)

#facial recoginition data for my face
dani0 = face_recognition.load_image_file("dani0.jpg")
dani0_face_encoding = face_recognition.face_encodings(dani0)[0]

dani1 = face_recognition.load_image_file("dani1.jpg")
dani1_face_encoding = face_recognition.face_encodings(dani1)[0]

dani2 = face_recognition.load_image_file("dani2.jpg")
dani2_face_encoding = face_recognition.face_encodings(dani2)[0]

known_face_encodings = [
    dani0_face_encoding,
    dani1_face_encoding,
    dani2_face_encoding
]
known_face_names = [
    "dani0",
    "dani1",
    "dani2"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#facial rec identifying me
i=0

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    i=i+1
    if i%100==0:
        i=0
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            now = strftime("%H:%M:%S", gmtime())
            print(now)

            #if it matching me then start features
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                video_capture.release()
                cv2.destroyAllWindows()

                #say hello
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text="Hello, Dani")
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                with open("output.mp3", "wb") as out:
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                playsound('/home/dani/Desktop/jar/output.mp3')


                #tell me a quote from my list of quotes
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=ranQuote)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                with open("output.mp3", "wb") as out:
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                playsound('/home/dani/Desktop/jar/output.mp3')

                #tell me my list of things i have to do
                con = sl.connect('todo.db')
                data = con.execute("select task from todo where completed = 'n';")
                with con:
                    for row in data:
                        rawTasks = row
                        tasks = ''.join(rawTasks)
                        print(tasks)
                client = texttospeech.TextToSpeechClient()
                synthesis_input = texttospeech.SynthesisInput(text=tasks)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                with open("output.mp3", "wb") as out:
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                playsound('/home/dani/Desktop/jar/output.mp3')

                #starting voice search
                url = 'google.com'
                RATE = 16000
                CHUNK = int(RATE / 10)  # 100ms

                #opens microphone for live voice listening
                class MicrophoneStream(object):
                    def __init__(self, rate, chunk):
                        self._rate = rate
                        self._chunk = chunk

                        self._buff = queue.Queue()
                        self.closed = True

                    def __enter__(self):
                        self._audio_interface = pyaudio.PyAudio()
                        self._audio_stream = self._audio_interface.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self._rate,
                            input=True,
                            frames_per_buffer=self._chunk,
                            stream_callback=self._fill_buffer,
                        )

                        self.closed = False

                        return self

                    def __exit__(self, type, value, traceback):
                        self._audio_stream.stop_stream()
                        self._audio_stream.close()
                        self.closed = True
                        self._buff.put(None)
                        self._audio_interface.terminate()

                    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
                        self._buff.put(in_data)
                        return None, pyaudio.paContinue

                    def generator(self):
                        while not self.closed:
                            chunk = self._buff.get()
                            if chunk is None:
                                return
                            data = [chunk]

                            while True:
                                try:
                                    chunk = self._buff.get(block=False)
                                    if chunk is None:
                                        return
                                    data.append(chunk)
                                except queue.Empty:
                                    break

                            yield b"".join(data)

                #jarva's voice commands live in here
                def listen_print_loop(responses):
                    num_chars_printed = 0
                    for response in responses:
                        if not response.results:
                            continue

                        result = response.results[0]
                        if not result.alternatives:
                            continue

                        transcript = result.alternatives[0].transcript
                        overwrite_chars = " " * (num_chars_printed - len(transcript))

                        if not result.is_final:
                            sys.stdout.write(transcript + overwrite_chars + "\r")
                            sys.stdout.flush()
                            num_chars_printed = len(transcript)

                        else:
                            print_state = transcript + overwrite_chars
                            print(print_state)

                            #voice search command
                            if re.search(r"\b(please search)\b", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()
                                synthesis_input = texttospeech.SynthesisInput(text="searching for "+transcript[14:])
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                with open("output.mp3", "wb") as out:
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                playsound('/home/dani/Desktop/jar/output.mp3')

                                #voice search command
                                search_string = transcript[14:]
                                search_string = search_string.replace(' ', '+', )
                                browser = webdriver.Chrome('chromedriver')
                                for i in range(1):
                                    matched_elements = browser.get("https://www.google.com/search?q=" + search_string + "&start=" + str(i))

                            #exit program voice command
                            if re.search(r"\b(we are done here)\b", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()
                                synthesis_input = texttospeech.SynthesisInput(text="ok. Goodbye, love ya")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                with open("output.mp3", "wb") as out:
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                playsound('/home/dani/Desktop/jar/output.mp3')
                                break

                            #task completed voice command
                            if re.search(r"mark task complete", transcript, re.I):
                                taskVoice0 = transcript[19:]
                                taskVoice1 = taskVoice0.lower()
                                taskVoice = taskVoice1.replace(" ", "")
                                print(taskVoice)
                                con = sl.connect('todo.db')
                                with con:
                                    data = con.execute(f"update todo set completed='y' where task='{taskVoice}';")
                                for row in data:
                                    print(row)

                                client = texttospeech.TextToSpeechClient()
                                synthesis_input = texttospeech.SynthesisInput(text="task "+transcript[19:]+" is now complete")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                with open("output.mp3", "wb") as out:
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                playsound('/home/dani/Desktop/jar/output.mp3')

                            num_chars_printed = 0

                            #play chess voice command
                            if re.search(r"\b(do you want to play a game)\b", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()
                                synthesis_input = texttospeech.SynthesisInput(text="yea. let me load up chess")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                with open("output.mp3", "wb") as out:
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                playsound('/home/dani/Desktop/jar/output.mp3')



                def main():
                    language_code = "en-US"

                    client = speech.SpeechClient()
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=RATE,
                        language_code=language_code,
                    )

                    streaming_config = speech.StreamingRecognitionConfig(
                        config=config, interim_results=True
                    )

                    with MicrophoneStream(RATE, CHUNK) as stream:
                        audio_generator = stream.generator()
                        requests = (
                            speech.StreamingRecognizeRequest(audio_content=content)
                            for content in audio_generator
                        )

                        responses = client.streaming_recognize(streaming_config, requests)

                        listen_print_loop(responses)

                if __name__ == "__main__":
                    main()

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
