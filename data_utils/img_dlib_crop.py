import face_recognition

img = face_recognition.load_image_file('../data/res/sample/demo1.jpg')
locations = face_recognition.face_locations(img)
print(locations)