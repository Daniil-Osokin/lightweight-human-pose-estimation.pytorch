import argparse

import torch

from human_pose_estimator.models.with_mobilenet import PoseEstimationWithMobileNet
from human_pose_estimator.modules.load_state import load_state


def convert_to_onnx(net, output_name):
    input = torch.randn(1, 3, 256, 456)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)
