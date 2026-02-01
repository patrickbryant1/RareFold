#############ARGS#############
BASE=. #Change this depending on your local path
ID=1FU0
DATADIR=$BASE/data/predict_test_case/$ID
FASTAFILE=$DATADIR/$ID'.fasta'
NUM_REC=3 #Number of recycles to use for the prediction
PARAMS=$BASE/data/params/params50000.npy
OUTDIR=$BASE/data/predict_test_case/$ID'/'
# The sequence has to be defined with the noncanonical amino acids (NCAAs) in
# threeletter code separated by hyphens, and the regular in oneletter e.g.
# >1FU0_A
# MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLK-SEP-IMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA
#
# We also need a fasta where the NCAA is "X" for the MSA search
# >1FU0_A
# MEKKEFHIVAETGIHARPATLLVQTASKFNSDINLEYKGKSVNLKXIMGVMSLGVGQGSDVTITVDGADEAEGMAAIVETLQKEGLA

FASTAWITHX=$DATADIR/$ID'_X.fasta'

#########Step1: Create MSA with HHblits#########
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
#MSA
MSA=$DATADIR/$ID'.a3m'

#Write individual fasta files for all unique sequences
if test -f $MSA; then
	echo $MSA exists
else
	$BASE/hh-suite/build/bin/hhblits -i $FASTAWITHX -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $MSA
fi


#########Step2: Make MSA features#########
#Here we also use "FASTAWITHX", the NCAA frames and features are mapped
#later in the predict script when the MSA is sampled
MSA_FEATS=$OUTDIR/msa_features.pkl
if test -f $MSA_FEATS; then
	echo $MSA_FEATS exists
else
	python3 $BASE/src/make_msa_seq_feats.py --input_fasta_path $FASTAWITHX \
	--input_msas $MSA --outdir $OUTDIR
fi

#########Step3: Predict#########
#The NCAAs will be saved to chain B in the prediction for easier visualisation
python3 ./src/predict_sc.py --predict_id $ID \
--MSA_feats $MSA_FEATS \
--fasta $FASTAFILE \
--num_recycles $NUM_REC \
--params $PARAMS \
--outdir $OUTDIR
