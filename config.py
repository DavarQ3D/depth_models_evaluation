import argparse
from custom_assets.datasets import Dataset
from custom_assets.models import Model, AlignmentType

parser = argparse.ArgumentParser(description='Depth estimation model evaluation')

parser.add_argument('--dataset',    default=Dataset.IPHONE.name, choices=[e.name for e in Dataset])
parser.add_argument('--model',      default=Model.Torch_UNIDEPTH_V2.name, choices=[e.name for e in Model])
parser.add_argument('--align',      action='store_true')
parser.add_argument('--alignType',  default=AlignmentType.MedianBased.name, choices=[e.name for e in AlignmentType])
parser.add_argument('--alignShift', action='store_true')
