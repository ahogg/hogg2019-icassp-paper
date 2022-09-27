from s4d.utils import *
from s4d.diar import Diar
from s4d import viterbi
from s4d.segmentation import init_seg
from s4d.segmentation import segmentation
from s4d.segmentation import bic_linear
from s4d.clustering import hac_bic
from sidekit.sidekit_io import init_logging
from s4d.gui.dendrogram import plot_dendrogram

# set the logger
loglevel = logging.INFO
init_logging(level=loglevel)

#  set the input audio (EN2002a_D01-01_UNKNOWN meeting) or mfcc file and the speech activity detection file (optional).
# data_dir = 'AMI/'
data_dir = '/Users/aidanhogg/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Documents/AMI/'
show = 'EN2002a_D01-01_UNKNOWN'
input_show = os.path.join(data_dir, 'audio', show + '.wav')
input_sad = None

#  size of left or right windows
win_size = 250

# threshold for: linear segmentation, BIC HAC and Viterbi
thr_l = 2
thr_h = 3
thr_vit = -250

# if save_all is True then all produced diarization are saved
save_all = True

# prepare various variables
wdir = os.path.join('out', show)

if not os.path.exists(wdir):
    os.makedirs(wdir)

# extract and load the MFCC
logging.info('Make MFCC')

if save_all:
    fe = get_feature_extractor(input_show, type_feature_extractor='basic')
    mfcc_filename = os.path.join(wdir, show + '.mfcc.h5')
    fe.save(show, output_feature_filename=mfcc_filename)
    fs = get_feature_server(mfcc_filename, feature_server_type='basic')
else:
    fs = get_feature_server(input_show, feature_server_type='basic')

cep, _ = fs.load(show)

# The initial diarization is loaded from a speech activity detection diarization (SAD)
# or a segment is created from the first to the last MFCC feature.
logging.info('Check initial segmentation')

if input_sad is not None:
    init_diar = Diar.read_seg(input_sad)
    init_diar.pack(50)
else:
    init_diar = init_seg(cep, show)

if save_all:
    init_filename = os.path.join(wdir, show + '.i.seg')
    Diar.write_seg(init_filename, init_diar)

# First segmentation: Segment each segment of init_diar using the Gaussian Divergence method
logging.info('Gaussian Divergence segmentation')

seg_diar = segmentation(cep, init_diar, win_size)

if save_all:
    seg_filename = os.path.join(wdir, show + '.s.seg')
    Diar.write_seg(seg_filename, seg_diar)

# This segmentation over the signal fuses consecutive segments of the same speaker from the start to the end of
# the record. The measure employs the ΔBIC based on Bayesian Information Criterion , using full covariance Gaussians.
logging.info('Linear BIC, alpha: %f', thr_l)

bicl_diar = bic_linear(cep, seg_diar, thr_l, sr=False)

if save_all:
    bicl_filename = os.path.join(wdir, show + '.l.seg')
    Diar.write_seg(bicl_filename, bicl_diar)

# Perform a BIC HAC
logging.info('BIC HAC, alpha: %f', thr_h)

bic = hac_bic.HAC_BIC(cep, bicl_diar, thr_h, sr=False)
bich_diar = bic.perform(to_the_end=True)

if save_all:
    bichac_filename = os.path.join(wdir, show + '.h.seg')
    Diar.write_seg(bichac_filename, bich_diar)

link, data = plot_dendrogram(bic.merge, 0, size=(25, 6), log=True)

# re-segmentation
logging.info('Viterbi decoding, penalties: %f', thr_vit)
vit_diar = viterbi.viterbi_decoding(cep, bich_diar, thr_vit)

if save_all:
    vit_filename = os.path.join(wdir, show + '.d.seg')
    Diar.write_seg(vit_filename, vit_diar)
