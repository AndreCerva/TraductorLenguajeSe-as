import cv2,os #cv2 para procesamiento de imagens, os para trabajar con el sistema operativo
import azure.cognitiveservices.speech as speechsdk #servicio de azure para speech and text
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient#Libreria de Azure para Custom Vision service
from msrest.authentication import ApiKeyCredentials#Libreria de Azure para autenticar credenciales de servicio
from dotenv import load_dotenv #Trabajar con variables de entorno

load_dotenv() #cargar las variables de entorno
PREDICTIONKEY=os.getenv("PREDICTIONKEY")#Key del prediction custom vision
ENDPOINT = os.getenv("ENDPOINT")#enpoint del servicio
IDPROJECT=os.getenv("IDPROJECT")#Id de nuestro proyecto
SPEECHSUBS=os.getenv("SPEECHSUBS")#servicio de speech
REGIONSPEECH=os.getenv("REGIONSPEECH")#region creamos el servicio de speech

credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTIONKEY})#La clave para hacer uso del servicio de custom vision desplegado
predictor = CustomVisionPredictionClient(ENDPOINT, credentials)#Endpoint donde se encuentra nuestro servicio de custom vision desplegado
speech_config = speechsdk.SpeechConfig(subscription=SPEECHSUBS, region=REGIONSPEECH)#Inicializar el servicio de speech
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)#configuramos el servicio de speech

speech_config.speech_synthesis_voice_name='es-MX-DaliaNeural' #Lenguaje en el que se hablara
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)#configuracion del speech

def main():
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)#abrir la cámara y completar la inicialización de la cámara
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)#Cambiamos el ancho de la imagen 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)#Cambiamos la altura de la imagen
    while (camera.isOpened()):#detectar si la inicialización se realizó correctamente
        ret, image = camera.read() #capturar fotogramas, image devuelve el fotograma capturado, ret true si es éxitoso la captura
        if ret== True:#Si se ha capturado correctamente el fotograma
            cv2.imshow('video',image)#Mostrar la serie de fotogramas que se están tomando
            if cv2.waitKey(1) & 0xFF == ord('t'):#Si se preciona la tecla: t , entra al condicional
                cv2.imwrite('capture.png', image)#Se guarda el fotograma cuando se presiono la tecla
                camera.release()#Apaga la camara
                cv2.destroyAllWindows()#Destruir todas las ventanas que se hayan mostrado anteriormente con cv2
                break#Sale del ciclo while
        else: #En caso de no poder capturar un fotograma
            print('Error al intentar ingresar a la camara')#Mostrar posible error de pq no se capturo fotograma
    with open("capture.png", mode="rb") as captured_image: #Abre la imagen guardada en formato de lectura
        results = predictor.detect_image(IDPROJECT,"I4",captured_image)#Se envia la imagen tomada con el id y el nombre del proyecto
    for prediction in results.predictions:#Se miran todas las predicciones hechas por Azure almacenadas en results cuando enviamos la imagen
        if prediction.probability > 0.9:#Para todas las predicciones que se hayan tenido que sean mayores a un 90% de seguridad
            print(prediction)#Se muestran todos los detalles de la predicción 
            text=prediction.tag_name#se almacena en una variable el tag name de la prediccion
            speech_synthesizer.speak_text_async(text).get()#Se envia el texto a hablar
            bbox = prediction.bounding_box#Cuadros delimetadores que se obtienen de la predicción
            #Para los cuadros delimitadores, hacemos un cálculo simple basado en el tamaño de la imagen, establecemos el color del cuadro delimitador y el grosor del borde. 
            #Dibujamos estos cuadros delimetadores en la imagen
            result_image = cv2.rectangle(image, (int(bbox.left * 640), int(bbox.top * 480)), (int((bbox.left + bbox.width) * 640), int((bbox.top + bbox.height) * 480)),(137,87,35), 3)
            cv2.putText(image,f" {prediction.tag_name} ({round(prediction.probability*100,2)})",(int(bbox.left * 630)+60, int(bbox.top * 470)-25),1, 1,(255,46,0),2)
            cv2.imwrite('capture.png', result_image)#Se guarda la imagen que se mando al servicio de custom vision
    imagen = cv2.imread('capture.png') #se lee la imagen con el nombre de la imagen guardada
    cv2.imshow('Object detection',imagen)#se muestra la imagen ya con el recuadro del objecto detectado
    cv2.waitKey(0) #espera a que se presione una tecla para cerrar la ventana
    cv2.destroyAllWindows() #destruye todas las ventanas que se hayan mostrado anteriormente con cv2

if __name__ == '__main__': #Si se ejecuta el archivo como script
    main() #Se ejecuta el método main