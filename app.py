import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy
import google.generativeai as genai
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")


col1,col2 =st.columns([2,1])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
        
        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        print(fingers1)
        return fingers1, lmList1
    else:
        return None

def draw(info, img, prev_pos,canvas):
    fingers, lmlist = info
    current_pos = None
    if fingers == [0,1,0,0,0]:
        current_pos = lmlist[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255,0,255),10)
    elif fingers == [1,0,0,0,0]:
        canvas = numpy.zeros_like(img)
    return current_pos,canvas



# Continuously get frames from the webcam


def sendtoAI(model,canvas,fingers):
    if fingers == [1,1,1,1,0]:
        pil_image = Image.fromarray(canvas)

        response = model.generate_content(["Solve this math problem - if you are not able to see a maths problem just write i am not able to understand", pil_image])
        return response.text



prev_pos = None
canvas = None
image_combines = None

output_text=""



while True:

    success,img = cap.read()
    img = cv2.flip(img,1)
    if canvas is None:
        canvas = numpy.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList1 = info
        print(fingers)
        prev_pos,canvas = draw(info, img, prev_pos,canvas)
        output_text = sendtoAI(model,canvas,fingers)


    image_combines = cv2.addWeighted(img, 0.7,canvas,0.3,0)
    FRAME_WINDOW.image(image_combines,channels="BGR")
    if output_text:
        output_text_area.text(output_text)
    # cv2.imshow("Image",img)
    # cv2.imshow("Canvas",canvas)
    # cv2.imshow("Image Combined",image_combines)
    cv2.waitKey(1)