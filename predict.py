import cv2 as cv
import torch
import torchvision.transforms as transforms
from setup import predict, model
import argparse

def main(args):
    # classes
    classes = ['0000', '1000', '5000', '20000', '50000']

    # create module
    money_model = model()

    # model load weight
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    money_model.load_state_dict(torch.load('money_weight.pth', map_location=device))
    money_model.eval()

    # classification and show in video
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    cap = cv.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        image = frame.copy()
        image = cv.resize(image, (224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.Normalize(mean, std)
        ])
        image = transform(image).unsqueeze(0)
        acc, predict_class = predict(money_model, image)
        if acc >= args.thread_hold and predict_class != 0:

            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv.putText(frame, classes[predict_class], org, font,
                       fontScale, color, thickness, cv.LINE_AA)

        cv.imshow('camera', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--thread-hold', type=float, default=0.6)
    args = parser.parse_args()
    main(args)






