import requests
import json
import speech_recognition as sr
import pyttsx3

API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 220)
tts_engine.setProperty("volume", 0.9)

recognizer = sr.Recognizer()

def speak(text):
    print(f"Robot: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text
        except sr.WaitTimeoutError:
            speak("No response detected. That’s suspicious.")
            return "NO_RESPONSE"
        except sr.UnknownValueError:
            speak("Didn’t catch that. Could you say it again?")
            return listen()
        except sr.RequestError:
            speak("Trouble hearing you. Let’s try again.")
            return ""

def ask_groq(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    messages = conversation_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

speak("Security Check: Let’s chat for a bit!")
conversation = [{"role": "system", "content": "You’re a security guard patrolling a neighborhood, asking natural, adaptive questions to determine if someone is suspicious. Start with a casual opener and adjust based on their answers."}]
initial_question = "Hey, what brings you out in the neighborhood this late?"
speak(initial_question)

responses = []
max_turns = 10

for turn in range(max_turns):
    user_answer = listen()
    if user_answer == "NO_RESPONSE":
        speak("You’re being suspicious with that silence!")
        print("\nSecurity Verdict: Suspicious")
        break
    if not user_answer:
        continue
    responses.append(user_answer)
    conversation.append({"role": "user", "content": user_answer})

    if turn >= 2:
        verdict_prompt = (
            "Here’s the conversation so far:\n" +
            "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
            "\nAs a security guard, decide if this person is suspicious based on evasiveness, contradictions, or hostility. "
            "If you have enough info, say 'Verdict: Suspicious' or 'Verdict: Not Suspicious' and stop asking. "
            "If not, ask another natural follow-up question."
        )
        next_step = ask_groq(verdict_prompt, conversation)
        
        if "Verdict:" in next_step:
            if "Suspicious" in next_step:
                speak("You’re being suspicious!")
            else:
                speak("Looks like you’re all good!")
            print(f"\nSecurity Verdict: {next_step}")
            break
        else:
            speak(next_step)
            conversation.append({"role": "assistant", "content": next_step})
    else:
        follow_up_prompt = (
            "Based on this conversation:\n" +
            "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
            "\nAsk a natural, adaptive follow-up question to probe further about their presence in the neighborhood."
        )
        next_question = ask_groq(follow_up_prompt, conversation)
        speak(next_question)
        conversation.append({"role": "assistant", "content": next_question})

if turn == max_turns - 1:
    final_verdict = ask_groq(
        "Conversation:\n" +
        "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
        "\nIs this person suspicious in the neighborhood? Say 'Suspicious' or 'Not Suspicious'."
    )
    if "Suspicious" in final_verdict:
        speak("You’re being suspicious!")
    else:
        speak("Looks like you’re all good!")
    print(f"\nSecurity Verdict: {final_verdict}")