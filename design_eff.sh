#############ARGS#############
BASE=. #Change this depending on your local path
ID=1ssc
DATADIR=$BASE/data/design_test_case/$ID
REC_FASTA=$DATADIR/$ID'_receptor.fasta'
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08

#########Step1: Create MSA with HHblits#########
#MSA
REC_MSA=$DATADIR/$ID'_receptor.a3m'
if test -f $REC_MSA; then
	echo $REC_MSA exists
else
	$BASE/hh-suite/build/bin/hhblits -i $REC_FASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $REC_MSA
fi


#############PARAMS FOR DESIGN#############
MAX_RECYCLES=8 #max_recycles (default=3)
BINDER_LENGTHS="10,11,12"
NITER=1000
RESAMPLE_FREQ=100 #How often to resample the MSA
BATCH_SIZE=5 #The batch size determines how many design threads are run simultaneously
PARAMS=$BASE/data/params/finetuned_params25000.npy
RARE_AAS="MSE,MLY,PTR,SEP,TPO,MLZ,ALY,HIC,HYP,M3L,PFF,MHO" #Specify the threeletter code for the NCAA you want to use for design
#You can also design with the regular AAs by specifying e.g. only ALA (only the NCAAs will be added onto the regular for design)
#All possible AAs:
#'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
#'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
#'TYR', 'VAL', 'MSE', 'TPO', 'MLY', 'CME', 'PTR', 'SEP',
#'SAH', 'CSO', 'PCA', 'KCX', 'CAS', 'CSD', 'MLZ', 'OCS', 'ALY',
#'CSS', 'CSX', 'HIC', 'HYP', 'YCM', 'YOF', 'M3L', 'PFF', 'CGU',
# 'FTR', 'LLP', 'CAF', 'CMH', 'MHO'

CYCLIC=True #Set to False if you want to design linear binders
MAX_WORKERS=18
OUTDIR=$DATADIR/

#######Step2: Make MSA features#######
MSA_FEATS=$OUTDIR/msa_features.pkl
if test -f $MSA_FEATS; then
	echo $MSA_FEATS exists
else
  python3 $BASE/src/make_msa_seq_feats.py --input_fasta_path $REC_FASTA \
  --input_msas $REC_MSA --outdir $OUTDIR
fi


#######Step3: Design#######
#All metrics will be saved to "metrics.csv" in outdir
#If the run stops for some reason - don't worry - it will continue where it left off
python3 $BASE/src/mc_design_length_var_batch.py --predict_id $ID \
--MSA_feats $MSA_FEATS \
--num_recycles $MAX_RECYCLES \
--binder_lengths $BINDER_LENGTHS \
--num_iterations $NITER \
--resample_every_n $RESAMPLE_FREQ \
--batch_size $BATCH_SIZE \
--params $PARAMS \
--rare_AAs $RARE_AAS \
--cyclic_offset $CYCLIC \
--max_workers $MAX_WORKERS \
--outdir $OUTDIR

#The design process can result in clashes - especially in the cyclic case.
#We recommend running relax, but don't provide this for modified residues

#######Convert the design metrics to human readable format#######
python3 $BASE/src/convert_design_metrics_hr.py --raw_metrics $OUTDIR/metrics.csv \
--outdir $OUTDIR
