import speech_recognition as sr

r=sr.Recognizer()
mic = sr.Microphone(sample_rate=12000)
with mic as source:
     r.adjust_for_ambient_noise(source,duration=1)
     audio = r.record(source, duration=1)
     text= r.recognize_google(audio, language = 'en-IN', show_all=True)
     print(text)
