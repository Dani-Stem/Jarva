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


ranQuote = random.choice(quotes.quotesList)


video_capture = cv2.VideoCapture(0)


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

i=0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    i=i+1
    if i%100==0:
        i=0
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            now = strftime("%H:%M:%S", gmtime())
            print(now)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                client = texttospeech.TextToSpeechClient()

                # Set the text input to be synthesized
                synthesis_input = texttospeech.SynthesisInput(text="Hello, Dani")

                # Release handle to the webcam
                video_capture.release()
                cv2.destroyAllWindows()

                # Build the voice request, select the language code ("en-US") and the ssml
                # voice gender ("neutral")
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )

                # Select the type of audio file you want returned
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )

                # Perform the text-to-speech request on the text input with the selected
                # voice parameters and audio file type
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )

                with open("output.mp3", "wb") as out:
                    # Write the response to the output file.
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                    playsound('/home/dani/Desktop/jar/output.mp3')

                # Instantiates a client
                client = texttospeech.TextToSpeechClient()

                # Set the text input to be synthesized
                synthesis_input = texttospeech.SynthesisInput(text=ranQuote)

                # Build the voice request, select the language code ("en-US") and the ssml
                # voice gender ("neutral")
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )

                # Select the type of audio file you want returned
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )

                # Perform the text-to-speech request on the text input with the selected
                # voice parameters and audio file type
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                # The response's audio_content is binary.
                with open("output.mp3", "wb") as out:
                    # Write the response to the output file.
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                    playsound('/home/dani/Desktop/jar/output.mp3')

                # Instantiates a client
                client = texttospeech.TextToSpeechClient()

                # Set the text input to be synthesized

                con = sl.connect('todo.db')

                data = con.execute("select task from todo where completed = 'n';")

                with con:
                    for row in data:
                        rawTasks = row
                        tasks = ''.join(rawTasks)
                synthesis_input = texttospeech.SynthesisInput(text=tasks)

                # Build the voice request, select the language code ("en-US") and the ssml
                # voice gender ("neutral")
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )

                # Select the type of audio file you want returned
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                )

                # Perform the text-to-speech request on the text input with the selected
                # voice parameters and audio file type
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                # The response's audio_content is binary.
                with open("output.mp3", "wb") as out:
                    # Write the response to the output file.
                    out.write(response.audio_content)
                    print('Audio content written to file "output.mp3"')
                    playsound('/home/dani/Desktop/jar/output.mp3')

                url = 'google.com'

                # Audio recording parameters
                RATE = 16000
                CHUNK = int(RATE / 10)  # 100ms


                class MicrophoneStream(object):
                    """Opens a recording stream as a generator yielding the audio chunks."""

                    def __init__(self, rate, chunk):
                        self._rate = rate
                        self._chunk = chunk

                        # Create a thread-safe buffer of audio data
                        self._buff = queue.Queue()
                        self.closed = True

                    def __enter__(self):
                        self._audio_interface = pyaudio.PyAudio()
                        self._audio_stream = self._audio_interface.open(
                            format=pyaudio.paInt16,
                            # The API currently only supports 1-channel (mono) audio
                            # https://goo.gl/z757pE
                            channels=1,
                            rate=self._rate,
                            input=True,
                            frames_per_buffer=self._chunk,
                            # Run the audio stream asynchronously to fill the buffer object.
                            # This is necessary so that the input device's buffer doesn't
                            # overflow while the calling thread makes network requests, etc.
                            stream_callback=self._fill_buffer,
                        )

                        self.closed = False

                        return self

                    def __exit__(self, type, value, traceback):
                        self._audio_stream.stop_stream()
                        self._audio_stream.close()
                        self.closed = True
                        # Signal the generator to terminate so that the client's
                        # streaming_recognize method will not block the process termination.
                        self._buff.put(None)
                        self._audio_interface.terminate()

                    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
                        """Continuously collect data from the audio stream, into the buffer."""
                        self._buff.put(in_data)
                        return None, pyaudio.paContinue

                    def generator(self):
                        while not self.closed:
                            # Use a blocking get() to ensure there's at least one chunk of
                            # data, and stop iteration if the chunk is None, indicating the
                            # end of the audio stream.
                            chunk = self._buff.get()
                            if chunk is None:
                                return
                            data = [chunk]

                            # Now consume whatever other data's still buffered.
                            while True:
                                try:
                                    chunk = self._buff.get(block=False)
                                    if chunk is None:
                                        return
                                    data.append(chunk)
                                except queue.Empty:
                                    break

                            yield b"".join(data)


                def listen_print_loop(responses):
                    """Iterates through server responses and prints them.

                    The responses passed is a generator that will block until a response
                    is provided by the server.

                    Each response may contain multiple results, and each result may contain
                    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
                    print only the transcription for the top alternative of the top result.

                    In this case, responses are provided for interim results as well. If the
                    response is an interim one, print a line feed at the end of it, to allow
                    the next result to overwrite it, until the response is a final one. For the
                    final one, print a newline to preserve the finalized transcription.
                    """
                    num_chars_printed = 0
                    for response in responses:
                        if not response.results:
                            continue

                        # The `results` list is consecutive. For streaming, we only care about
                        # the first result being considered, since once it's `is_final`, it
                        # moves on to considering the next utterance.
                        result = response.results[0]
                        if not result.alternatives:
                            continue

                        # Display the transcription of the top alternative.
                        transcript = result.alternatives[0].transcript

                        # Display interim results, but with a carriage return at the end of the
                        # line, so subsequent lines will overwrite them.
                        #
                        # If the previous result was longer than this one, we need to print
                        # some extra spaces to overwrite the previous result
                        overwrite_chars = " " * (num_chars_printed - len(transcript))

                        if not result.is_final:
                            sys.stdout.write(transcript + overwrite_chars + "\r")
                            sys.stdout.flush()

                            num_chars_printed = len(transcript)

                        else:
                            print_state = transcript + overwrite_chars
                            print(print_state)

                            if re.search(r"\b(please search)\b", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()

                                # Set the text input to be synthesized
                                synthesis_input = texttospeech.SynthesisInput(text="searching for "+transcript[14:])

                                # Build the voice request, select the language code ("en-US") and the ssml
                                # voice gender ("neutral")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )

                                # Select the type of audio file you want returned
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )

                                # Perform the text-to-speech request on the text input with the selected
                                # voice parameters and audio file type
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                # The response's audio_content is binary.
                                with open("output.mp3", "wb") as out:
                                    # Write the response to the output file.
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                    playsound('/home/dani/Desktop/jar/output.mp3')

                                search_string = transcript[14:]
                                search_string = search_string.replace(' ', '+', )
                                browser = webdriver.Chrome('chromedriver')
                                for i in range(1):
                                    matched_elements = browser.get("https://www.google.com/search?q=" + search_string + "&start=" + str(i))

                            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()

                                # Set the text input to be synthesized
                                synthesis_input = texttospeech.SynthesisInput(text="Goodbye")

                                # Build the voice request, select the language code ("en-US") and the ssml
                                # voice gender ("neutral")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )

                                # Select the type of audio file you want returned
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )

                                # Perform the text-to-speech request on the text input with the selected
                                # voice parameters and audio file type
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                # The response's audio_content is binary.
                                with open("output.mp3", "wb") as out:
                                    # Write the response to the output file.
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                    playsound('/home/dani/Desktop/jar/output.mp3')
                                break

                            if re.search(r"mark task zero as complete|mark task 0 as complete", transcript, re.I):
                                client = texttospeech.TextToSpeechClient()

                                # Set the text input to be synthesized
                                synthesis_input = texttospeech.SynthesisInput(text="Task zero is now complete")

                                # Build the voice request, select the language code ("en-US") and the ssml
                                # voice gender ("neutral")
                                voice = texttospeech.VoiceSelectionParams(
                                    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                                )

                                # Select the type of audio file you want returned
                                audio_config = texttospeech.AudioConfig(
                                    audio_encoding=texttospeech.AudioEncoding.MP3
                                )

                                # Perform the text-to-speech request on the text input with the selected
                                # voice parameters and audio file type
                                response = client.synthesize_speech(
                                    input=synthesis_input, voice=voice, audio_config=audio_config
                                )
                                # The response's audio_content is binary.
                                with open("output.mp3", "wb") as out:
                                    # Write the response to the output file.
                                    out.write(response.audio_content)
                                    print('Audio content written to file "output.mp3"')
                                    playsound('/home/dani/Desktop/jar/output.mp3')

                            num_chars_printed = 0


                def main():
                    # See http://g.co/cloud/speech/docs/languages
                    # for a list of supported languages.
                    language_code = "en-US"  # a BCP-47 language tag

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

                        # Now, put the transcription responses to use.
                        listen_print_loop(responses)


                if __name__ == "__main__":
                    main()




            face_names.append(name)


    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





