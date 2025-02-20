from typing import Dict, Callable, Optional
from gtts import gTTS
from playsound3 import playsound
import speech_recognition as sr
import webbrowser
import os


def create_tts_audio(text: str, filename: str) -> None:
    """
    Generate audio file from text using gTTS.

    :param text: Input text to convert to speech.
    :param filename: Output filename for the audio.
    """
    tts = gTTS(text=text, lang='en')
    tts.save(filename)


def play_audio(filename: str) -> None:
    """
    Play generated audio file.

    :param filename: Path to audio file to play.
    """
    playsound(filename)


def text_to_speech(text: str) -> None:
    """
    Convert text to speech and play it immediately.

    :param text: Text to speak aloud.
    """
    temp_file = "temp_speech.mp3"
    create_tts_audio(text, temp_file)
    play_audio(temp_file)
    os.remove(temp_file)


def speech_to_text(timeout: int = 5) -> Optional[str]:
    """
    Convert speech from microphone input to text.

    :param timeout: Maximum wait time for input (seconds).
    :return: Recognized text or None if failed.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            return recognizer.recognize_google(audio).lower()
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except Exception as e:
            print(f"Recognition error: {e}")
            return None


def open_wikipedia() -> None:
    """
    Execute 'open wikipedia' command
    """
    webbrowser.open("https://www.wikipedia.org")


def get_commands() -> Dict[str, Callable[[], None]]:
    """
    Return available voice commands mapping
    """
    return {
        'open wikipedia': open_wikipedia,
        'launch wikipedia': open_wikipedia,
        'wikipedia': open_wikipedia,
    }


def process_command(text: str, commands: Dict[str, Callable[[], None]]) -> bool:
    """
    Process recognized text against registered commands.

    :param text: Recognized speech text.
    :param commands: Dictionary of command phrases and functions.

    :return: True if command was executed, False otherwise
    """
    for phrase, action in commands.items():
        if phrase in text:
            action()
            return True
    return False


def main() -> None:
    """
    Entry point.
    """
    commands = get_commands()
    text_to_speech("Voice command system ready")

    while True:
        recognized = speech_to_text()

        if not recognized:
            continue

        print(f"Recognized: {recognized}")

        if 'exit' in recognized:
            text_to_speech("Goodbye")
            break

        if process_command(recognized, commands):
            text_to_speech("Command executed")
        else:
            text_to_speech("Command not recognized")


if __name__ == "__main__":
    main()