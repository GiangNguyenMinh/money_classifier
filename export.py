import torch
import torch.onnx
from setup import model
import logging
import onnx
import argparse

logging.basicConfig(level=logging.INFO)

def export_torchscript(model, img, file):
    try:
        logging.info('Starting export model to torchscript ...')
        traced_script_model = torch.jit.trace(model, img)
        traced_script_model.save(file)
    except Exception as e:
        logging.info('export failure: {}'.format(e))


def export_onnx(model, img, file):
    try:
        torch.onnx.export(model,
                          img,
                          file,
                          export_params=True,
                          opset_version=10,
                          input_names=['input'],  # specify name of ip in ep module,the ip have same dimensions with exp ip
                          output_names=['output'],  # specify name of op in ep module, have the same dimensions as exp op
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                          verbose=False)
        # checks
        model_onnx = onnx.load(file)
        onnx.checker.check_model(model_onnx)

        onnx.save(model_onnx, file)
    except Exception as e:
        logging.info('export failure: {}'.format(e))

def export_tensorrt():
    pass

def run(args):
    file = {'script_file': './money.torchscript',
            'onnx_file': './money.onnx'}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    money_model = model()
    logging.info('create model success')
    money_model.load_state_dict(torch.load('money_weight.pth', map_location=device))
    money_model.eval()
    img = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(2):
        y = money_model(img)
        logging.info('run dry')

    if args.torchscript:
        export_torchscript(money_model, img, file['script_file'])
    if args.onnx:
        export_onnx(money_model, img, file['onnx_file'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export modle')
    parser.add_argument('--onnx', action='store_true', help='export to onnx')
    parser.add_argument('--torchscript', action='store_true', help='export to torchscript')
    args = parser.parse_args()
    run(args)







