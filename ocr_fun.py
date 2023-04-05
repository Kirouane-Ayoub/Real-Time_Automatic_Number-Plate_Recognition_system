import easyocr
import pytesseract
import cv2

def pytesseract_fun(img) :
    # you can add here some preprocessing 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pytesseract.image_to_string(img_rgb)
    return result

def easyocr_fun(img) : 
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    return result[0][-2]