import argparse
import torch

from PIL import Image
from torchvision import transforms

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
parser.add_argument('input', type=str, help='path to input image')

def show_graph_recursively(script):
    print(script.graph)
    for child in script.children():
        show_graph_recursively(child)

def _infer(path_to_checkpoint_file, path_to_input_image):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cpu()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0).cpu()

        print(images.shape)

        model.eval()
        # m = torch.jit.trace(model, images)
        # show_graph_recursively(m)
        # torch.jit.save(m, "shvn-torch.pt")

        length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(images)
        length_prediction = length_logits.max(1)[1]
        digit1_prediction = digit1_logits.max(1)[1]
        digit2_prediction = digit2_logits.max(1)[1]
        digit3_prediction = digit3_logits.max(1)[1]
        digit4_prediction = digit4_logits.max(1)[1]
        digit5_prediction = digit5_logits.max(1)[1]

        print('length:', length_prediction.item())
        print('digits:', digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item(), digit5_prediction.item())


def main(args):
    path_to_checkpoint_file = args.checkpoint
    path_to_input_image = args.input

    _infer(path_to_checkpoint_file, path_to_input_image)


if __name__ == '__main__':
    main(parser.parse_args())
