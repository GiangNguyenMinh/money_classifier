import torch
import onnxruntime
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms

ort_session = onnxruntime.InferenceSession("./money.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infernce_ort(img):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def main():
    classes = ['0000', '1000', '5000', '20000', '50000']
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    cap = cv.VideoCapture(0)
    start = cv.getTickCount()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        image = frame.copy()
        image = cv.resize(image, (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean, std)
        ])
        image = transform(image).unsqueeze(0)
        out_onnx = infernce_ort(image)
        pred = classes[np.argmax(out_onnx[0])]
        fps = cv.getTickFrequency() / (cv.getTickCount()-start)
        start = cv.getTickCount()
        cv.putText(frame, 'fps: {}'.format(str(int(fps))), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv.putText(frame, pred, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv.imshow('onnx', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()







