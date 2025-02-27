import os
import openai
import speech_recognition as sr
import pyttsx3
import requests
import json
import numpy as np
from PIL import Image
import cv2

# set api key
openai.api_key = ""
free_key = ""
gpt_url = "https://api.pawan.krd/v1/chat/completions"
dall_url = "https://api.pawan.krd/v1/images/generations"
headers = {"Authorization": f"Bearer {free_key}"}

# set gpt to act as asked
messages = [{"role": "system", "content": "Act like an intelligent assistant"}]

# initialize speech recognition
r = sr.Recognizer()

# initialize text to speech
engine = pyttsx3.init()
engine.setProperty('rate', 140)

def text2speech(message):
	engine.say(message)
	engine.runAndWait()


def get_command(source):
	r.adjust_for_ambient_noise(source)
	 
	audio = r.listen(source)

	content = r.recognize_google(audio)
	content = content.lower()

	print(f"command: {content}")
	return content


def url2img(url):
	img = Image.open(requests.get(url, stream=True).raw)
	opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
	return opencv_img


def image_generator(message):
	if message:
		res = requests.post(
			url=dall_url,
			json={
				"response_format": "b64_json",
				"prompt": message,
				"n": 1,
				"size": "1024x1024"
			},
			headers=headers
		).json()
		img_url = res["data"][0]["url"]
		img_final = url2img(img_url)
		return img_final


def ask_gpt(message):
	if message:
		messages.append({"role": "user", "content": message})
		print(messages)
		gpt = requests.post(
			url=gpt_url,
			json={
				"model": "gpt-3.5-turbo",
				"messages": messages
			},
			headers=headers
		).json()
		res = gpt["choices"][0]["message"]["content"]
		print(f"ChatGPT: {res}")
		messages.append({"role": "assistant", "content": res})

		return res


def main():
	# infinite loop to detect if a user speak
	while(1):
		print("listening...")
		try:
			# use the microphone as source for input.
			with sr.Microphone() as source:
				command = get_command(source)
			if "exit command" in command:
				print("----------------------logs---------------------")
				print(json.dumps(messages, indent=2))
			if "generate" in command:
				print("processing...")
				image = image_generator(command.lower().replace("generate an", "").strip())
				cv2.imshow('dall-e', image)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			else:
				print("processing...")
				gpt_res = ask_gpt(command)
				text2speech(gpt_res)
		except sr.RequestError as e:
			print(f"Could not request results: {e}")
		except sr.UnknownValueError:
			print("An error occurred, can you please say that again?")

if __name__ == "__main__":
	main()
