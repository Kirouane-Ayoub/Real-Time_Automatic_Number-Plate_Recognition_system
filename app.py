from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st
import easyocr
import pytesseract

def pytesseract_fun(img) :
    # you can add here some preprocessing 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pytesseract.image_to_string(img_rgb)
    return result

def easyocr_fun(img) : 
    # you can add here some preprocessing
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    return result[0][-2]

with st.sidebar : 
    st.image("icon.png" , width=300)
    select_type_detect = st.selectbox("Detection from :  ",
                                            ("File", 
                                             "Live"))
    save_crops = st.selectbox("Do you want to save Crops ? " , 
                             ("Yes" , "No"))
    select_device = st.selectbox("Select compute Device :  ",
                                            ("CPU", "GPU"))
    save_output_video = st.radio("Save output video?",
                                            ('Yes', 'No'))

    confd = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)

tab0 , tab1 = st.tabs(["Home" , "Detection"])
with tab0:
    st.header("About MY Project : ")
    st.image("ANPR.jpg" , width=600)
    st.write("""Real-Time Automatic Number Plate Recognition (ANPR) system is an advanced technology that combines the power of 
    computer vision and deep learning to recognize and interpret license plate information from vehicles in real-time. 
    This system is built using the state-of-the-art YOLOv8 object detection algorithm and Optical Character Recognition (OCR) technology. 
    The system can detect and read number plates from live video feeds or recorded footage from cameras, making it an ideal solution for traffic management,
    law enforcement, and parking management applications. With its high accuracy, speed, and real-time capabilities, 
    this system is changing the way number plate recognition is performed, making it more efficient and reliable than ever before.""")


with tab1 : 
    if select_device == "GPU" : 
        DEVICE_NAME = st.selectbox("Select GPU index : " , 
                                     (0, 1 , 2)) 
    elif select_device =="CPU" : 
        DEVICE_NAME = "cpu"
    fpsReader = cvzone.FPS()
    class_names = ["d_license_plate", "pl_license_plate"]
    if select_type_detect == "File" : 
        file = st.file_uploader("Select Your File : " ,
                                 type=["mp4" , "mkv"])
        if file : 
            source = file.name
            cap = cv2.VideoCapture(source)
    elif select_type_detect == "Live" : 
        source = st.text_input("Past Your Url here and Click Entre")
        cap = cv2.VideoCapture(source)
    # creat the model
    model = YOLO("num_plate_recv1.pt")
    frame_window = st.image( [] )
    start , stop = st.columns(2)
    with start : 
        start = st.button("Click To Start")
    with stop : 
        stop = st.button("Click To Stop" , key="ghedqLKHF")
    if start :
        while True :
            _ , img = cap.read() 
            if save_output_video == "Yes" :
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
                fourcc = cv2.VideoWriter_fourcc(*'MP4V') #use any fourcc type to improve quality for the saved video
                out = cv2.VideoWriter(f'results/{source.split(".")[0]}.mp4', fourcc, 10, (w, h)) #Video settings
            # fps counter
            fps, img = fpsReader.update(img,pos=(20,50),
                                        color=(0,255,0),
                                        scale=2,thickness=3)
            # make the prediction 
            results = model(img ,conf=confd ,
                             iou=iou,
                             device=DEVICE_NAME)
            for result in results : 
                # depackage results
                bboxs = result.boxes 
                for box in bboxs : 
                    # bboxes
                    x1  , y1 , x2 , y2 = box.xyxy[0]
                    x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
                    # confidence 
                    conf = math.ceil((box.conf[0] * 100 ))
                    # class name
                    clsi = int(box.cls[0])
                    # calculate the width and the height
                    w,h = x2 - x1 , y2 - y1
                    # convert it into int
                    w , h = int(w) , int(h)
                    # draw our bboxes 
                    cvzone.cornerRect(img ,(x1 , y1 , w , h) ,l=7)
                    
                    crop = img[y1 : y1+h , x1:x1+w]
                    text_result = easyocr_fun(crop)

                    #text_result = pytesseract_fun(crop)

                    #put text_result inside our image
                    cvzone.putTextRect(img , f"{text_result}" ,
                                       (max(0,x1) , max(20 , y1)),
                                       thickness=1 ,colorR=(0,0,255) ,
                                       scale=0.9 , offset=3)
                    if save_crops == "Yes" : 
                        cv2.imwrite(f'crops/{text_result}.jpg', crop)
                    else : 
                        pass
                    try: 
                        out.write(img)
                    except:
                        pass

            frame  = cv2.cvtColor( img , 
                                cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
    else:
        try:
            cap.release()
        except : 
            pass
