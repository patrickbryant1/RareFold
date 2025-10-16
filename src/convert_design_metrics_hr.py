import pandas as pd
import glob
import numpy as np
import sys
from ast import literal_eval
import argparse
import pdb



parser = argparse.ArgumentParser(description = """Convert the design metrics to human readable format.""")

parser.add_argument('--raw_metrics', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with raw design metrics.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


#######################MAIN#######################

#Parse args
args = parser.parse_args()
raw_metrics = pd.read_csv(args.raw_metrics[0])
outdir = args.outdir[0]

#Merge all metrics into row-wise format for human readability
if_dist_binder = np.array([x for x in raw_metrics.if_dist_binder.apply(literal_eval)])
plddts = np.array([x for x in raw_metrics.plddt.apply(literal_eval)])
inter_clash_frac = np.array([x for x in raw_metrics.inter_clash_frac.apply(literal_eval)])
intra_clash_frac = np.array([x for x in raw_metrics.intra_clash_frac.apply(literal_eval)])
losses = np.array([x for x in raw_metrics.loss.apply(literal_eval)])
seqs = np.array([x for x in raw_metrics.sequence.apply(literal_eval)])
int_seqs = np.array([x for x in raw_metrics.int_seq.apply(literal_eval)])

#Assign new df
compiled_df = {'iter':[], 'replicate':[], 'length':[], 'if_dist_binder':[], 'plddt':[], 'inter_clash_frac':[], 'intra_clash_frac':[], 'loss':[], 'sequence':[], 'int_seq':[]}
for i in range(if_dist_binder.shape[1]):
    compiled_df['iter'].extend([*raw_metrics.iteration.values])
    compiled_df['replicate'].extend([i]*if_dist_binder.shape[0])
    compiled_df['length'].extend([len(seqs[0,i])]*if_dist_binder.shape[0])
    compiled_df['if_dist_binder'].extend([*if_dist_binder[:,i]])
    compiled_df['plddt'].extend([*plddts[:,i]])
    compiled_df['inter_clash_frac'].extend([*inter_clash_frac[:,i]])
    compiled_df['intra_clash_frac'].extend([*inter_clash_frac[:,i]])
    compiled_df['loss'].extend([*losses[:,i]])
    compiled_df['sequence'].extend([*seqs[:,i]])
    compiled_df['int_seq'].extend([*int_seqs[:,i]])

compiled_df = pd.DataFrame.from_dict(compiled_df)
compiled_df.to_csv(outdir+'metrics_hr.csv',index=None)
