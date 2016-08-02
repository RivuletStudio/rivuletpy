from rivuletpy.trace import trace
from rivuletpy.utils.io import *
import argparse

parser = argparse.ArgumentParser(description='Arguments for see anisotropic filters.')
parser.add_argument('--file', type=str, default=None, help='The file to filter')
parser.add_argument('--threshold', type=float, default=0, help='threshold to distinguish the foreground and background')

parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=False)

parser.add_argument('--length', type=int, default=4, help='branches with lengths below this threshold will be abandoned')

parser.add_argument('--radius', dest='radius', action='store_true')
parser.add_argument('--no-radius', dest='radius', action='store_false')
parser.set_defaults(radius=False)

parser.add_argument('--ske', dest='ske', action='store_true')
parser.add_argument('--no-ske', dest='ske', action='store_false')
parser.set_defaults(ske=True)

parser.add_argument('--response_as_speed', dest='response_as_speed', action='store_true')
parser.add_argument('--no-response_as_speed', dest='response_as_speed', action='store_false')
parser.set_defaults(response_as_speed=False)

parser.add_argument('--clean', dest='clean', action='store_true')
parser.add_argument('--no-clean', dest='clean', action='store_false')
parser.set_defaults(clean=True)

parser.add_argument('--original_image', type=str, default=None, help='The original image to get the soma radius')
parser.add_argument('--soma_threshold', type=float, default=None, help='The threshold on the original image to get soma radius')
args = parser.parse_args()

img = loadimg(args.file)
oimg = loadimg(args.original_image)
swc = trace(img, threshold=args.threshold, render=args.render, 
            length=args.length, ignore_radius= not args.radius,
            skedt=args.ske, coverage=.99, clean=args.clean, 
            response_as_speed = args.response_as_speed, 
            original_image=oimg,
            soma_threshold=args.soma_threshold)

saveswc('_'.join([args.file, str(args.threshold), str(args.length), str(args.radius), str(args.ske), str(args.clean), '.swc']), swc)
