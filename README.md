# Face-Mask-Detection
# Report

     
 For the Visea Innovative Information Technologies -> Work Report            
Bu Python programlama çalışmasında, Keras, Tensorflow, MobileNet ve OpenCV kullanarak bir Yüz Maskesi Dedektörü oluşturdum. Bunun bir WebCam’a bağlanarak yaptım. Daha fazla iyileştirme ile bu tür modeller, insanları maskesiz tespit etmek ve tanımlamak için CCTV kameralarla entegre edilebilir.

Yüz maskesi detektörü, herhangi bir Morphed maskeli görüntü veri kümesi kullanmadı. Model doğrudur ve MobileNetV2 mimarisi kullanıldığından, hesaplama açısından da etkilidir ve dolayısıyla modeli gömülü sistemlere (Raspberry Pi, Google Coral, vb.) dağıtmayı kolaylaştırır.

Dolayısıyla bu sistem, Covid-19 salgını nedeniyle güvenlik amacıyla yüz maskesi tespiti gerektiren gerçek zamanlı uygulamalarda kullanılabilir. Bu proje, kamu güvenliği kurallarının takip edilmesini sağlamak için havalimanlarında, tren istasyonlarında, ofislerde, okullarda ve halka açık yerlerde uygulama için gömülü sistemlerle entegre edilebilir.

Yazılan kod ta her method’un ne işlev yaptığı aşağıda satır satır belirtilmiştir.

Gerekli olan kütüphaneler:
tensorflow>=1.15.2
keras==2.3.1
imutils==0.5.3
numpy==1.18.2
opencv-python==4.2.0.*
matplotlib==3.2.1
scipy==1.4.1







# Öncelikle gerekli paketleri import etmemiz gerekiyor.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# maskeyi algıla ve tahmin etme için gerekli method.
def detect_and_predict_mask(frame, faceNet, maskNet):
	
              # Frame’in  boyutlarını yakalama ve ardından bir blob oluşturma
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# blob'u ağ üzerinden geçirme ve face detectionları alma işlemi.
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Face list’i , bunlara karşılık gelen konumları 
              # ve yüz maskesi ağımızdaki tahminlerin  listesini başlatma işlemi.
	
	faces = []
	locs = []
	preds = []

	#  detection ların üzerinden geçme işlemi yapıldı.
	for i in range(0, detections.shape[2]):
	              # Detection ile ilişkili güveni ortaya çıkarma
		confidence = detections[0, 0, i, 2]

		# Confidence’in minimum confidence den daha yüksek olmasını sağlayarak zayıf algılamaları filtreleme işlemi yapıldı.
		if confidence > 0.5:
			# nesnenin sınırlayıcı kutusunun (x, y) koordinatlarını hesaplama			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# sınırlayıcı kutuların frame’in boyutları dahilinde olmasını sağlama işlemi			              
                                           (startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Yüz ROI'sini extract etme, BGR'den RGB kanalına dönüştürme
			# Order etme, 224x224 olarak yeniden boyutlandırma ve proprocess etme
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Yüzü ve sınırlayıcı kutuları ilgili alanlara ekleme.
			# list’ler 
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	

# En az bir yüz algılandıysa bir tahmin yapma işlemi
if len(faces) > 0:
		
                 # Daha hızlı çıkarım için * all * için toplu tahminler yapıldı.
                 # tek tek tahminler yerine aynı anda Face’ler işlendi.
                 # yukarıdaki "for" döngüsünde.

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# yüz konumlarının ve bunlara karşılık gelen konumların 2-tuple’yi döndürür.

	return (locs, preds)

# Serilazed edilmiş Face Detector modelimizi diskten yükleme işlemi.
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Face Mask  Detector  modelini diskten yükleme işlemi.
maskNet = load_model("mask_detector.model")

# Video akışını initialize etme işlemi.
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# video akışındaki frame’ler üzerinde döngü yapma işlemidir.
while True:
	# Frame’i thread edilen video akışından alma ve yeniden boyutlandırma işlemi.
	# maksimum 400 piksel genişliğe sahip olmak için yapılır.
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Frame’deki yüzleri tespit etme ve yüz maskesi takıp takmadıklarını belirleme işlemi.
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Algılanan yüz konumlarını ve bunlara karşılık gelen konumlar üzerinde döngü oluşturma işlemidir.
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

	# Bounding Box’ı  ve Text’i çizmek için kullanacağımız sınıf etiketini ve rengini belirleme.
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Probolity’i Label’a ekleme işlemi 
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Output frame’inde,  label’ı  ve bounding box dikdörtgenini görüntüleme.
		
                           cv2.putText(frame, label, (startX, startY - 10),
	             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Çıktı Frame’ini gösterme satırı.
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# "q" tuşuna basıldıysa döngüden çıkılır.
	if key == ord("q"):
	break

# Temizlik işlemi yapar.
cv2.destroyAllWindows()
vs.stop()


                                                                                                            
                                                                                                               Name: Hüseyin 
                                                                                                            Surname: Gülçiçek




![maskeli](https://user-images.githubusercontent.com/33606081/112515944-cccd6600-8da7-11eb-9ea8-69ff2530e4ab.PNG)
![maskesiz](https://user-images.githubusercontent.com/33606081/112515946-ce972980-8da7-11eb-8e6f-cd02b71b8e07.PNG)




