import os  
from dotenv import load_dotenv  
import io  
import azure.cognitiveservices.speech as speechsdk  
from openai import OpenAI
import time  
import datetime  
import threading  
import json, ast  

import requests  
from io import BytesIO  
import tempfile  
import numpy as np  
#import pyaudio  
  
load_dotenv("voice1.env")  
client = OpenAI(api_key=os.environ["api_key"], base_url=os.environ["base_url"])
Azure_speech_key = os.environ["Azure_speech_key"]  
Azure_speech_region = os.environ["Azure_speech_region"]  
Azure_speech_speaker = os.environ["Azure_speech_speaker"]  
WakeupWord = os.environ["WakeupWord"]  
WakeupModelFile = os.environ["WakeupModelFile"]  

messages = []  

# Set up Azure Speech-to-Text and Text-to-Speech credentials  
speech_key = Azure_speech_key  
service_region = Azure_speech_region  
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)  
# Set up Azure Text-to-Speech language  
speech_config.speech_synthesis_language = "zh-CN"  
# Set up Azure Speech-to-Text language recognition  
speech_config.speech_recognition_language = "zh-CN"  
lang = "zh-CN"  
# Set up the voice configuration  
speech_config.speech_synthesis_voice_name = Azure_speech_speaker  
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)  
connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)  
connection.open(True)  
# Creates an instance of a keyword recognition model. Update this to  
# point to the location of your keyword recognition model.  
model = speechsdk.KeywordRecognitionModel(WakeupModelFile)  
# The phrase your keyword recognition model triggers on.  
keyword = WakeupWord  
# Set up the audio configuration  
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)  
# Create a speech recognizer and start the recognition  
#speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)  
auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["ja-JP", "zh-CN"])  
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config,  
                                               auto_detect_source_language_config=auto_detect_source_language_config)  
unknownCount = 0  
sysmesg = {"role": "system", "content": os.environ["sysprompt_zh-CN"]}  
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]


isListenning=False
def display_text(s):
    print(s)
def speech_to_text():  
    global unknownCount  
    global lang,isListenning  
    print("Please say...")  
    result = speech_recognizer.recognize_once_async().get()  
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:  
        unknownCount = 0  
        isListenning=False
        return result.text  
    elif result.reason == speechsdk.ResultReason.NoMatch:  
        isListenning=False
        unknownCount += 1  
        error = os.environ["sorry_" + lang]  
        text_to_speech(error)  
        return '...'  
    elif result.reason == speechsdk.ResultReason.Canceled:  
        isListenning=False
        return "speech recognizer canceled." 
    

def getVoiceSpeed():  
    return 17  
  
def text_to_speech(text, _lang=None):  
    global lang  
    try:  
        result = buildSpeech(text).get()  
        #result = speech_synthesizer.speak_ssml_async(ssml_text).get()  
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  
            print("Text-to-speech conversion successful.")  
            return "Done."  
        else:  
            print(f"Error synthesizing audio: {result}")  
            return "Failed."  
    except Exception as ex:  
        print(f"Error synthesizing audio: {ex}")  
        return "Error occured!"  
        
def buildSpeech(text, _lang=None):
    voice_lang = lang  
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural"  
    ssml_text = f'''  
        <speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="{lang}"><voice name="{voice_name}"><lang xml:lang="{voice_lang}"><prosody rate="{getVoiceSpeed()}%">{text.replace('*', ' ')}</prosody></lang></voice></speak>  
    '''  
    print(f"{voice_name} {voice_lang}!")  
    return speech_synthesizer.speak_ssml_async(ssml_text)
  
def generate_text(prompt):  
    global messages  
    messages.append({"role": "user", "content": prompt})  
    #tools = getTools()  
    collected_messages = []
    last_tts_request = None

    split=True
    result=''
    
    response_gen = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages[-20:],
        stream=True
    )
    # 循环接收返回的message
    for chunk in response_gen:
        if chunk:
            chunk_message =  chunk.choices[0].delta.content  # 抽取流式message里的内容
            if chunk_message is not None:
                collected_messages.append(chunk_message)  # 收集保存message
                result=result+chunk_message
                
                if chunk_message in tts_sentence_end and split: # 发现分段标记：句子结束
                    text = ''.join(collected_messages).strip() # 构建句子
                    if len(text)>20: #如果这句足够长，后面不再分段朗读了
                        split=False
                    elif len(text)<6: #如果这句太短了，不单独朗读，并入后面
                        continue
                    if text != '': # 如果句子里只有 \n or space则跳过朗读
                        print(f"Speech synthesized to speaker for: {text}")
                        
                        last_tts_request = buildSpeech(text)
                        #last_tts_request = speech_synthesizer.speak_text_async(text)
                        collected_messages.clear()
                        
    messages.append({"role": "assistant", "content": result})
    
    if len(collected_messages)>0:
        text = ''.join(collected_messages).strip() 
        if text != '': 
            print(f"Speech synthesized to speaker for: {text}")
            last_tts_request = buildSpeech(text)
            collected_messages.clear()

    
    if last_tts_request:
        last_tts_request.get()        
    return result
  

def recognized_cb(evt):  
    result = evt.result  
    if result.reason == speechsdk.ResultReason.RecognizedKeyword:  
        print("RECOGNIZED KEYWORD: {}".format(result.text))  
    global done  
    done = True  

def canceled_cb(evt):  
    result = evt.result  
    if result.reason == speechsdk.ResultReason.Canceled:  
        print('CANCELED: {}'.format(result.cancellation_details.reason))  
    global done  
    done = True  
   
def start_recognition():  
    global unknownCount ,isListenning 
    while True:  
        keyword_recognizer = speechsdk.KeywordRecognizer()  
        keyword_recognizer.recognized.connect(recognized_cb)  
        keyword_recognizer.canceled.connect(canceled_cb) 
        first=os.environ["welcome_" + lang]
        display_text(first)  
        text_to_speech(first)  
        isListenning=True
        result_future = keyword_recognizer.recognize_once_async(model)  
        while True:  
            result = result_future.get()
            # Read result audio (incl. the keyword).
            if result.reason == speechsdk.ResultReason.RecognizedKeyword:
                print("Keyword recognized")
                isListenning=False
                break
            time.sleep(0.1)  
            
        display_text("很高兴为您服务，我在听请讲。")  
        text_to_speech("很高兴为您服务，我在听请讲。")  
        
          
        while unknownCount < 2:
            isListenning=True
            user_input = speech_to_text()
            
            print(f"You: {user_input}")  
            
            display_text(f"You: {user_input}")  
            response = generate_text(user_input)  
            #text_to_speech(f"你说的是：{user_input}")  
            
          
        bye_text = os.environ["bye_" + lang]  
        display_text(bye_text)  
        text_to_speech(bye_text)  
        
        unknownCount = 0  
        time.sleep(0.1)  

if __name__ == "__main__":
    start_recognition()