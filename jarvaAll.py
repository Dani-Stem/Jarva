from __future__ import print_function, division
import random
import quotes
import face_recognition
import cv2
import numpy as np
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
import sys
import sqlite3 as sl
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QInputDialog, QLineEdit
from playsound import playsound
from google.cloud import texttospeech

#connecting to db for life app
con = sl.connect('lifeApp.db')

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

                #life app w/ gui
                with con:
                    currDate0 = con.execute("select date('now', 'localtime')")
                    for value in currDate0:
                        currDate = value[0]

                with con:
                    currTime0 = con.execute("select time('now', 'localtime')")
                    for value in currTime0:
                        currTime = value[0]

                with con:
                    con.execute(
                        f"insert into life_app_main(date, time, all_goal, workout_yoga, rwd, hardware, software, veggies) values('{currDate}', '{currTime}', 'n', 'n', 'n', 'n', 'n', 'n');")

                    currId0 = con.execute("select last_insert_rowid()")
                    for value in currId0:
                        currId = value[0]
                        # print(currId)

                with con:
                    countAllGoals0 = con.execute(
                        f"select count(*) from life_app_main where date='{currDate}' and all_goal='y';")
                    for value in countAllGoals0:
                        countAllGoals = value[0]

                with con:
                    countWorkoutYoga0 = con.execute(
                        f"select count(*) from life_app_main where date='{currDate}' and workout_yoga='y';")
                    for value in countWorkoutYoga0:
                        countWorkoutYoga = value[0]

                with con:
                    countRwd0 = con.execute(f"select count(*) from life_app_main where date='{currDate}' and rwd='y';")
                    for value in countRwd0:
                        countRwd = value[0]

                with con:
                    countHardware0 = con.execute(
                        f"select count(*) from life_app_main where date='{currDate}' and hardware='y';")
                    for value in countHardware0:
                        countHardware = value[0]

                with con:
                    countSoftware0 = con.execute(
                        f"select count(*) from life_app_main where date='{currDate}' and software='y';")
                    for value in countSoftware0:
                        countSoftware = value[0]

                with con:
                    countVeggies0 = con.execute(
                        f"select count(*) from life_app_main where date='{currDate}' and veggies='y';")
                    for value in countVeggies0:
                        countVeggies = value[0]

                    if countVeggies != 0 and countSoftware != 0 and countRwd != 0 and countHardware != 0 and countWorkoutYoga != 0:
                        con.execute(f"update life_app_main set all_goal='y' where id={currId};")

                with con:
                    workoutYoga0 = con.execute(f"select workout_yoga from life_app_main where id={currId}")
                    for value in workoutYoga0:
                        workoutYoga = value[0]

                with con:
                    rwd0 = con.execute(f"select rwd from life_app_main where id={currId}")
                    for value in rwd0:
                        rwd = value[0]

                with con:
                    hardware0 = con.execute(f"select hardware from life_app_main where id={currId}")
                    for value in hardware0:
                        hardware = value[0]

                with con:
                    software0 = con.execute(f"select software from life_app_main where id={currId}")
                    for value in software0:
                        software = value[0]

                with con:
                    veggies0 = con.execute(f"select veggies from life_app_main where id={currId}")
                    for value in veggies0:
                        veggies = value[0]

                with con:
                    allGoals0 = con.execute(f"select all_goal from life_app_main where id={currId}")
                    for value in allGoals0:
                        allGoals = value[0]


                class App(QWidget):

                    def __init__(self):
                        super().__init__()
                        self.title = 'Life App'
                        self.left = 2000
                        self.top = 10
                        self.width = 640
                        self.height = 480
                        self.initUI()

                    def initUI(self):
                        self.setWindowTitle(self.title)
                        self.setGeometry(self.left, self.top, self.width, self.height)

                        self.feel()
                        self.recAct()

                        with con:
                            if countWorkoutYoga == 0:
                                self.workoutYoga()
                            else:
                                con.execute(f"update life_app_main set workout_yoga='y' where id={currId}")
                        with con:
                            if countRwd == 0:
                                self.rwd()
                            else:
                                con.execute(f"update life_app_main set rwd='y' where id={currId}")
                        with con:
                            if countHardware == 0:
                                self.hardware()
                            else:
                                con.execute(f"update life_app_main set hardware='y' where id={currId}")

                        with con:
                            if countSoftware == 0:
                                self.software()
                            else:
                                con.execute(f"update life_app_main set software='y' where id={currId}")

                        with con:
                            if countVeggies == 0:
                                self.veggies()
                            else:
                                con.execute(f"update life_app_main set veggies='y' where id={currId}")

                        self.nexAct()
                        self.show()
                        self.close()
                        with con:
                            data = con.execute("select * from life_app_main;")
                            for row in data:
                                print(row)
                        exit()

                    def feel(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="how do you feel")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "How do you feel: ", QLineEdit.Normal,
                                                               "")
                        with con:
                            if okPressed and text != '':
                                mood = text
                                con.execute(f"update life_app_main set mood='{mood}' where id={currId}")

                    def recAct(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="recent activity")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Most recent activity: ",
                                                               QLineEdit.Normal, "")
                        with con:
                            if okPressed and text != '':
                                recAct = text
                                con.execute(f"update life_app_main set recent_activity='{recAct}' where id={currId}")

                    def workoutYoga(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="workout or yoga")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Workout/Yoga: ", QLineEdit.Normal, "")
                        with con:
                            if okPressed and text != '':
                                workoutYoga = text
                                con.execute(f"update life_app_main set workout_yoga='{workoutYoga}' where id={currId}")

                    def rwd(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="Read Write draw")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Read/Write/Draw: ", QLineEdit.Normal,
                                                               "")
                        with con:
                            if okPressed and text != '':
                                rwd = text
                                con.execute(f"update life_app_main set rwd='{rwd}' where id={currId}")

                    def hardware(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="hardware")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Harwdware: ", QLineEdit.Normal, "")
                        with con:
                            if okPressed and text != '':
                                hardware = text
                                con.execute(f"update life_app_main set hardware='{hardware}' where id={currId}")

                    def software(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="software")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Software: ", QLineEdit.Normal, "")
                        with con:
                            if okPressed and text != '':
                                software = text
                                con.execute(f"update life_app_main set software='{software}' where id={currId}")

                    def veggies(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="veggie")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "veggies: ", QLineEdit.Normal, "")
                        with con:
                            if okPressed and text != '':
                                veggies = text
                                con.execute(f"update life_app_main set veggies='{veggies}' where id={currId}")

                    def nexAct(self):

                        client = texttospeech.TextToSpeechClient()
                        synthesis_input = texttospeech.SynthesisInput(text="Next activity")
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
                        playsound('/home/dani/Desktop/jar/output.mp3')

                        text, okPressed = QInputDialog.getText(self, "Get text", "Next activity: ", QLineEdit.Normal,
                                                               "")
                        with con:
                            if okPressed and text != '':
                                nexAct = text
                                con.execute(f"update life_app_main set next_activity='{nexAct}' where id={currId}")


                if __name__ == '__main__':
                    app = QApplication(sys.argv)
                    ex = App()
                    sys.exit(app.exec_())

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

                                board = board.Board.new()

                                # Returns a move object based on the users input. Does not check if the move is valid.
                                def get_user_move():
                                    # create a speech recognition object
                                    r = sr.Recognizer()

                                    # a function that splits the audio file into chunks
                                    # and applies speech recognition
                                    def get_large_audio_transcription(path):

                                        # open the audio file using pydub
                                        sound = AudioSegment.from_wav(path)
                                        # split audio sound where silence is 700 miliseconds or more and get chunks
                                        chunks = split_on_silence(sound,
                                                                  # experiment with this value for your target audio file
                                                                  min_silence_len=500,
                                                                  # adjust this per requirement
                                                                  silence_thresh=sound.dBFS - 14,
                                                                  # keep the silence for 1 second, adjustable as well
                                                                  keep_silence=500,
                                                                  )
                                        folder_name = "audio-chunks"
                                        # create a directory to store the audio chunks
                                        if not os.path.isdir(folder_name):
                                            os.mkdir(folder_name)
                                        whole_text = ""
                                        # process each chunk
                                        for i, audio_chunk in enumerate(chunks, start=1):
                                            # export audio chunk and save it in
                                            # the `folder_name` directory.
                                            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
                                            audio_chunk.export(chunk_filename, format="wav")
                                            # recognize the chunk
                                            with sr.AudioFile(chunk_filename) as source:
                                                audio_listened = r.record(source)
                                                # try converting it to text
                                                try:
                                                    text = r.recognize_google(audio_listened)
                                                except sr.UnknownValueError as e:
                                                    print("Error:", str(e))
                                                else:
                                                    text = f"{text.capitalize()}. "
                                                    # print(chunk_filename, ":", text)
                                                    whole_text += text
                                        # return the text for all chunks detected
                                        return whole_text

                                    with sr.Microphone() as source:
                                        # read the audio data from the default microphone

                                        print(board.to_string())

                                        client = texttospeech.TextToSpeechClient()
                                        synthesis_input = texttospeech.SynthesisInput(text="your move")
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
                                        playsound('/home/dani/Desktop/jar/output.mp3')

                                        print("Your Move: ")
                                        # sleep(2)
                                        audio_data = r.record(source, duration=5)
                                        # convert speech to text
                                        text = r.recognize_google(audio_data)
                                        move_str = text
                                        move_str = move_str.replace(" ", "")

                                    try:
                                        xfrom = letter_to_xpos(move_str[0:1])
                                        yfrom = 8 - int(move_str[
                                                        1:2])  # The board is drawn "upside down", so flip the y coordinate.
                                        xto = letter_to_xpos(move_str[2:3])
                                        yto = 8 - int(move_str[
                                                      3:4])  # The board is drawn "upside down", so flip the y coordinate.
                                        return ai.Move(xfrom, yfrom, xto, yto, False)
                                    except ValueError:

                                        client = texttospeech.TextToSpeechClient()
                                        synthesis_input = texttospeech.SynthesisInput(text="Invalid format")
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
                                        playsound('/home/dani/Desktop/jar/output.mp3')

                                        print("Invalid format")
                                        return get_user_move()

                                # Returns a valid move based on the users input.
                                def get_valid_user_move(board):
                                    while True:
                                        move = get_user_move()
                                        valid = False
                                        possible_moves = board.get_possible_moves(pieces.Piece.WHITE)
                                        # No possible moves
                                        if (not possible_moves):
                                            return 0

                                        for possible_move in possible_moves:
                                            if (move.equals(possible_move)):
                                                move.castling_move = possible_move.castling_move
                                                valid = True
                                                break

                                        if (valid):
                                            break
                                        else:
                                            client = texttospeech.TextToSpeechClient()
                                            synthesis_input = texttospeech.SynthesisInput(text="Invalid move")
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
                                            playsound('/home/dani/Desktop/jar/output.mp3')
                                            print("Invalid move.")
                                    return move

                                # Converts a letter (A-H) to the x position on the chess board.
                                def letter_to_xpos(letter):
                                    letter = letter.upper()
                                    if letter == 'A':
                                        return 0
                                    if letter == 'B':
                                        return 1
                                    if letter == 'C':
                                        return 2
                                    if letter == 'D':
                                        return 3
                                    if letter == 'E':
                                        return 4
                                    if letter == 'F':
                                        return 5
                                    if letter == 'G':
                                        return 6
                                    if letter == 'H':
                                        return 7

                                    raise ValueError("Invalid letter.")

                                #
                                # Entry point.

                                print(board.to_string())

                                while True:
                                    move = get_valid_user_move(board)
                                    if (move == 0):
                                        if (board.is_check(pieces.Piece.WHITE)):

                                            client = texttospeech.TextToSpeechClient()
                                            synthesis_input = texttospeech.SynthesisInput(text="Checkmate. Black Wins.")
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
                                            playsound('/home/dani/Desktop/jar/output.mp3')

                                            print("Checkmate. Black Wins.")
                                            break
                                        else:
                                            print("Stalemate.")

                                            client = texttospeech.TextToSpeechClient()
                                            synthesis_input = texttospeech.SynthesisInput(text="Stalemate")
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
                                            playsound('/home/dani/Desktop/jar/output.mp3')

                                            break

                                    board.perform_move(move)

                                    print("User move: " + move.to_string())
                                    print(board.to_string())

                                    ai_move = ai.AI.get_ai_move(board, [])
                                    if (ai_move == 0):
                                        if (board.is_check(pieces.Piece.BLACK)):

                                            client = texttospeech.TextToSpeechClient()
                                            synthesis_input = texttospeech.SynthesisInput(text="Checkmate. White wins")
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
                                            playsound('/home/dani/Desktop/jar/output.mp3')

                                            print("Checkmate. White wins.")

                                            break
                                        else:

                                            client = texttospeech.TextToSpeechClient()
                                            synthesis_input = texttospeech.SynthesisInput(text="Stalemate")
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
                                            playsound('/home/dani/Desktop/jar/output.mp3')

                                            print("Stalemate.")

                                            break

                                    board.perform_move(ai_move)
                                    print("AI move: " + ai_move.to_string())
                                    print(board.to_string())


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
