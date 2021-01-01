
from .yolov5.models.yolo import Model
import torch

class NoGPUAvailable(Exception):
    def __init__(self):
        print("GPU not available")

class Wrapper():
    def __init__(self, model_file):
        if not torch.cuda.is_available():
            raise NoGPUAvailable()

        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        self.model = Model("/include/object_detection/yolov5/models/yolov5s.yaml", ch=3, nc=4)
        self.model.load_state_dict(torch.load(model_file))

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        for im0s in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!

            # Padded resize
            # img = letterbox(im0s, new_shape=self.img_size)[0]

            # Convert
            img = im0s[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            
            img = torch.from_numpy(img).to(device)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=False)

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

             # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f'{n} {names[int(c)]}s, '  # add to string
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                print(cls)
                # label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)
                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)            

                # box, label, score = self.model.predict(img) # TODO you probably need to send the image to a tensor, etc.

                boxes.append(xyxy)
                labels.append(names[int(cls)])
                scores.append(conf)

        return boxes, labels, scores

# class Model():    # TODO probably extend a TF or Pytorch class!
#     def __init__(self):
#         # TODO Instantiate your weights etc here!
#         pass
#     # TODO add your own functions if need be!
