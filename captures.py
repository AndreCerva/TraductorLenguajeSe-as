"""Este codigo lo comentaremos en el taller con ayuda de Github Copilot =)"""
#take a set of photos with cv2
import cv2, os
camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#see the frames of the video to be captured
i=0
t=0
while(True):
    ret,frame=camera.read()
    cv2.imshow('frame',frame)
    key= cv2.waitKey(1) & 0xFF 
    if key == ord('i'):
        i+=1
        print(f"taking photo {i} for ILY")
        cv2.imwrite(f'.\images\ILV\ILY{i}.jpg',frame)
    if key== ord('t'):
        t+=1
        print(f"taking photo {t} for THX")
        cv2.imwrite(f'.\images\THX\THX{t}.jpg',frame)
    elif key== ord('q'):
        break
    