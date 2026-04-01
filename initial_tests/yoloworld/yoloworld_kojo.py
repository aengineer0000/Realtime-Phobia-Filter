from ultralytics import YOLOWorld
model = YOLOWorld('yolov8s-world.pt')

model.set_classes(['spider', 'bicycle'])

spider = 'spider.jpg'
bicycle = 'bicycle.jpg'

results = model.predict(
    source=spider,
    imgsz=320
    )
results[0].show()

results = model.predict(
    source=bicycle,
    imgsz=320
    )
results[0].show()


'''
Yoloworld Test Notes

- Lower resolution (320 instead of default 640) works better for our use case
- Does not seem to work on spiders or insects (will need to train a model to fill 
these holes)
- Inference timing is in the range of 40-80ms for a single image, giving us an expected 
FPS (given a video input), of about 10-20 FPS, which is currently too laggy for enjoyable 
video viewing
'''


