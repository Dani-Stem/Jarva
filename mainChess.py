import board0, pieces, ai
import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from playsound import playsound
from google.cloud import texttospeech

sr.energy_threshold = 600

def playChess():

    board = board0.Board.new()
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
            # sleep(5)

            print("Your Move: ")
            audio_data = r.record(source, duration=8)
            # convert speech to text
            text = r.recognize_google(audio_data)
            text0 = text.replace("your move", "")
            print(text)
            move_str = text0
            move_str = move_str.replace(" ", "")

        try:
            xfrom = letter_to_xpos(move_str[0:1])
            yfrom = 8 - int(move_str[1:2]) # The board is drawn "upside down", so flip the y coordinate.
            xto = letter_to_xpos(move_str[2:3])
            yto = 8 - int(move_str[3:4]) # The board is drawn "upside down", so flip the y coordinate.
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

