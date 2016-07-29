from filtering.anisotropic import * 
from rivuletpy.utils.io import * 
from scipy import io as sio
import skfmm
import argparse

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
   
parser = argparse.ArgumentParser(description='Arguments for see anisotropic filters.')
parser.add_argument('--file', type=str, default=None, help='The file to filter')
parser.add_argument('--rlow', type=float, default=1., help='The lower bound of radius to try')
parser.add_argument('--rhigh', type=float, default=2., help='The higher bound of radius to try')
parser.add_argument('--rstep', type=float, default=0.5, help='The step size of trying radius')
parser.add_argument('--rho', type=float, default=0.5, help='The step size of trying radius')
args = parser.parse_args()
  
if args.file is None:
	print('Usage: ./seefillter.py --file <PATH2FILE> --rlow <lowerbound> --rhigh <higherbound>, --rstep <step>, --rho <rho>')

img = loadimg(args.file)
    
radii = np.arange(args.rlow, args.rhigh, args.rstep)
rps, _, _ = response(img.astype('float'), rsptype='bg', radii=np.asarray(radii), rho=args.rho, memory_save=False)

# writetiff3d(''.join([args.file, '.', str(rlow), '.', str(rhigh), '.' , str(rstep), '.', str(rho), '.rps.tif'], rps))
img = nib.Nifti1Image(rps, np.eye(4))
nib.save(img, '_'.join([args.file, str(args.rlow), str(args.rhigh), str(args.rstep), str(args.rho), '.rps.nii.gz']))